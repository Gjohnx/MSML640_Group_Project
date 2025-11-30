import numpy as np
from typing import Tuple
from .base import DetectionMethod


class RandomDetectionMethod(DetectionMethod):
    
    def __init__(self):
        self.cube_colors = np.full((6, 3, 3), -1, dtype=np.int8)
    
    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Find all unknown tiles (where value is -1)
        unknown_indices = np.where(self.cube_colors == -1)
        
        if len(unknown_indices[0]) > 0:
            # Pick a random unknown tile
            idx = np.random.randint(0, len(unknown_indices[0]))
            face = unknown_indices[0][idx]
            row = unknown_indices[1][idx]
            col = unknown_indices[2][idx]
            
            # Assign a random color (0-5)
            self.cube_colors[face, row, col] = np.random.randint(0, 6)
        
        return frame, self.cube_colors.copy()

