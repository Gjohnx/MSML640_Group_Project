from typing import Tuple, Optional, List
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

from ..base import DetectionMethod


class CubeCNN(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.03),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.03),
            nn.MaxPool2d(2),
            nn.Conv2d(96, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.05),
            nn.MaxPool2d(2),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.05),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192 * 4 * 4, 384),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(384, 9 * 6),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        logits = self.classifier(x)
        return logits.view(-1, 9, 6)


class YOLOCNNDetectionMethod(DetectionMethod):
    
    def __init__(
        self,
        yolo_model_path: Optional[str] = None,
        cnn_model_path: Optional[str] = None,
        conf_threshold: float = 0.25,
        device: Optional[str] = None,
    ):
        root = Path(__file__).resolve().parent
        # YOLO weights default to best.pt beside this file
        self.yolo_model_path = Path(yolo_model_path) if yolo_model_path else root / "best.pt"
        # CNN weights default to cube_cnn.pt beside this file (same folder)
        default_cnn = root / "cube_cnn.pt"
        self.cnn_model_path = Path(cnn_model_path) if cnn_model_path else default_cnn
        self.conf_threshold = conf_threshold
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        if not self.yolo_model_path.exists():
            raise FileNotFoundError(f"YOLO model file not found: {self.yolo_model_path}")
        if not self.cnn_model_path.exists():
            raise FileNotFoundError(f"CNN model file not found: {self.cnn_model_path}")

        self.yolo = self._load_yolo()
        self.cnn = self._load_cnn()
        self.transform = self._build_transform()
        self.last_cube_colors = np.full((6, 3, 3), '?', dtype='<U2')

    def _load_yolo(self):
        try:
            return YOLO(str(self.yolo_model_path))
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model from {self.yolo_model_path}: {e}")

    def _load_cnn(self):
        model = CubeCNN().to(self.device)
        try:
            state = torch.load(self.cnn_model_path, map_location=self.device)
            model.load_state_dict(state, strict=True)
        except Exception as e:
            raise RuntimeError(f"Failed to load CNN model from {self.cnn_model_path}: {e}")
        model.eval()
        return model

    @staticmethod
    def _build_transform():
        # Matches training/inference preprocessing used in detect_colors.py
        return transforms.Compose(
            [
                transforms.Resize((100, 100)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def _crop_to_pil(self, frame_bgr: np.ndarray, box_xyxy: List[float]) -> Optional[Image.Image]:
        x1, y1, x2, y2 = map(int, box_xyxy)
        h, w = frame_bgr.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        # Convert BGR -> RGB for PIL/transforms
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        return Image.fromarray(crop_rgb)

    def _predict_labels(self, pil_img: Image.Image) -> List[int]:
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.cnn(tensor)
            preds = logits.argmax(dim=-1).squeeze(0).cpu().tolist()
        return preds

    @staticmethod
    def _labels_to_chars(preds: List[int]) -> List[str]:
        mapping = {0: "U", 1: "D", 2: "F", 3: "B", 4: "R", 5: "L"}
        return [mapping.get(p, "?") for p in preds]

    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Optional[Tuple[float, float, float]]]:
        if frame is None or frame.size == 0:
            return frame, np.full((6, 3, 3), '?', dtype='<U2'), None

        processed_frame = frame.copy()
        try:
            results = self.yolo.predict(
                source=frame,
                conf=self.conf_threshold,
                save=False,
                verbose=False
            )
        except Exception as e:
            print(f"YOLO inference error: {e}")
            return processed_frame, np.full((6, 3, 3), '?', dtype='<U2'), None

        # Start from the previously detected state so faces accumulate over time.
        cube_colors = self.last_cube_colors.copy()
        if not results:
            self.last_cube_colors = cube_colors.copy()
            return processed_frame, cube_colors, None

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            self.last_cube_colors = cube_colors.copy()
            return processed_frame, cube_colors, None

        selected_box = None
        for box in result.boxes:
            cls_id = int(box.cls[0])
            if cls_id != 0:
                continue
            selected_box = box
            break
        if selected_box is None:
            selected_box = result.boxes[0]

        xyxy = selected_box.xyxy[0].cpu().numpy().tolist()
        x1, y1, x2, y2 = map(int, xyxy)
        cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 4)

        pil_crop = self._crop_to_pil(frame, xyxy)
        if pil_crop is None:
            return processed_frame, cube_colors, None

        preds = self._predict_labels(pil_crop)
        chars = self._labels_to_chars(preds)
        print(f"YOLO+CNN predicted labels (ids): {preds}")
        print(f"YOLO+CNN predicted labels (chars): {chars}")

        # Update the appropriate face based on the center sticker
        try:
            face = np.array(chars, dtype='<U2').reshape(3, 3)
            center = face[1, 1]
            face_index_map = {"U": 0, "R": 1, "F": 2, "D": 3, "L": 4, "B": 5}
            target_idx = face_index_map.get(center, 2)  # default to Front if unknown
            cube_colors[target_idx] = face
        except Exception:
            pass

        # Store the latest detected cube colors
        self.last_cube_colors = cube_colors.copy()

        return processed_frame, cube_colors, None

    def reset(self):
        pass
