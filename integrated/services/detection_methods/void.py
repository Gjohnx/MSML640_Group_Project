import numpy as np
from typing import Tuple
from .base import DetectionMethod


class VoidDetectionMethod(DetectionMethod):
    
    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        cube_colors = np.full((6, 3, 3), -1, dtype=np.int8)
        return frame, cube_colors

