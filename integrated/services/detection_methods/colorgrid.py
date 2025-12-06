
from typing import Tuple, Optional, Dict
import numpy as np
import cv2

from .base import DetectionMethod

from ..vision.color_grid_detector import detect_color_grid


class ColorGridDetectionMethod(DetectionMethod):
    """DetectionMethod adapter around `detect_color_grid()`."""

    def __init__(
        self,
        expected_size: int = 300,          # warp size for the rectified face
        sample_ratio: float = 0.36,        # central sampling square ratio per cell
        min_area_ratio: float = 0.08,       # minimal face area ratio in the frame
        approx_epsilon_ratio: float = 0.02,     # epsilon ratio for polygon approximation
        ref_lab: Optional[Dict[str, Tuple[float, float, float]]] = None,
        ref_json_path: Optional[str] = None,
        debug: bool = True
    ):
        self.expected_size = expected_size
        self.sample_ratio = sample_ratio
        self.min_area_ratio = min_area_ratio
        self.approx_epsilon_ratio = approx_epsilon_ratio
        self.ref_lab = ref_lab
        self.ref_json_path = ref_json_path
        self.debug = debug

        # Cache last detected Front face to smooth out '?' flickers
        self._last_front = np.full((3, 3), '?', dtype='<U1')

    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Optional[Tuple[float, float, float]]]:
        """Run color-grid detection and adapt to (processed_frame, cube_colors, rotation)."""
        if frame is None or frame.size == 0:
            return frame, np.full((6, 3, 3), '?', dtype='<U1'), None

        # Call the robust detector from your previous codebase
        out = detect_color_grid(
            frame_bgr=frame,
            debug=self.debug,
            expected_size=self.expected_size,
            sample_ratio=self.sample_ratio,
            min_area_ratio=self.min_area_ratio,
            approx_epsilon_ratio=self.approx_epsilon_ratio,
            ref_lab=self.ref_lab,
            ref_json_path=self.ref_json_path
        )

        processed = out.get("overlay_bgr", frame)

        cube = np.full((6, 3, 3), '?', dtype='<U1')

        # If a face was detected, write it to the Front face (index 2)
        if out.get("ok") and out.get("grid") is not None:
            front = out["grid"].astype('<U1')

            # Smooth out '?' flickers using the last known face
            mask_unknown = (front == '?')
            if np.any(mask_unknown):
                front = front.copy()
                front[mask_unknown] = self._last_front[mask_unknown]
            self._last_front = front

            cube[2] = front

            # Annotate the processed image
            try:
                h, w = processed.shape[:2]
                cv2.putText(
                    processed, "Front face detected",
                    (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (36, 255, 12), 2, cv2.LINE_AA
                )
            except Exception:
                pass

        return processed, cube, None
    
    def reset(self):
        pass
