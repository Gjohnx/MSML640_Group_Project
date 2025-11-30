import numpy as np
from typing import Tuple
from .base import DetectionMethod


class RandomDetectionMethod(DetectionMethod):
    
    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        cube_colors = np.random.randint(0, 6, (6, 3, 3), dtype=np.int8)
        return frame, cube_colors

