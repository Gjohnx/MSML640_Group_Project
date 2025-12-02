import numpy as np
from typing import Tuple, Optional
from .base import DetectionMethod


class VoidDetectionMethod(DetectionMethod):
    
    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Optional[Tuple[float, float, float]]]:
        cube_colors = np.full((6, 3, 3), '?', dtype=str)
        return frame, cube_colors, None

