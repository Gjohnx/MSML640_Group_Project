from typing import Tuple, Optional
from pathlib import Path
import numpy as np
import cv2
from ultralytics import YOLO

from ..base import DetectionMethod


class YOLODetectionMethod(DetectionMethod):
    
    def __init__(self, model_path: Optional[str] = None, conf_threshold: float = 0.25):
        if model_path is None:
            # Use best.pt in the same directory as this file
            model_path = Path(__file__).parent / "best.pt"
        else:
            model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"YOLO model file not found: {model_path}")
        
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.model = None
        self._load_model()
    
    def _load_model(self):
        try:
            self.model = YOLO(str(self.model_path))
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model from {self.model_path}: {e}")
    
    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Optional[Tuple[float, float, float]]]:
        if frame is None or frame.size == 0:
            cube_colors = np.full((6, 3, 3), '?', dtype='<U1')
            return frame, cube_colors, None
        
        # Make a copy to avoid modifying the original
        processed_frame = frame.copy()
        
        # Run YOLO inference
        try:
            results = self.model.predict(
                source=frame,
                conf=self.conf_threshold,
                save=False,
                verbose=False
            )
        except Exception as e:
            # If inference fails, return original frame
            print(f"YOLO inference error: {e}")
            cube_colors = np.full((6, 3, 3), '?', dtype='<U1')
            return processed_frame, cube_colors, None
        
        # Get the first result
        if len(results) > 0:
            result = results[0]
            
            # Draw bounding boxes if detections exist
            if result.boxes is not None and len(result.boxes) > 0:
                # Draw simple bounding boxes filtered by class ID 0
                for box in result.boxes:
                    # Filter by class ID 0
                    cls_id = int(box.cls[0])
                    if cls_id != 0:
                        continue
                    
                    # Get box coordinates (xyxy format)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Confidence
                    conf = float(box.conf[0])
                    print(f"Detected object (class {cls_id}): box=({x1}, {y1}, {x2}, {y2}), confidence={conf:.3f}")
                    
                    # Draw rectangle
                    cv2.rectangle(
                        processed_frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        4
                    )
        
        # Return processed frame with bounding boxes
        cube_colors = np.full((6, 3, 3), '?', dtype='<U1')
        return processed_frame, cube_colors, None
    
    def reset(self):
        pass
