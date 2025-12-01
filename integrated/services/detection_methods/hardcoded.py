import numpy as np
from typing import Tuple, Optional
from .base import DetectionMethod


class HardcodedDetectionMethod(DetectionMethod):
    
    def __init__(self):
        # Hardcoded cube matrix (6 faces, 3 rows, 3 columns)
        # Colors: 0=white, 1=yellow, 2=red, 3=orange, 4=green, 5=blue
        self.hardcoded_cube = np.array([
            # Face 0 (White/Up)
            [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0]],
            # Face 1 (Yellow/Down)
            [[1, 1, 1],
             [1, 1, 1],
             [1, 1, 1]],
            # Face 2 (Red/Front)
            [[2, 2, 2],
             [2, 2, 2],
             [2, 2, 2]],
            # Face 3 (Orange/Back)
            [[3, 3, 3],
             [3, 3, 3],
             [3, 3, 3]],
            # Face 4 (Green/Right)
            [[4, 4, 4],
             [4, 4, 4],
             [4, 4, 4]],
            # Face 5 (Blue/Left)
            [[5, 5, 5],
             [5, 5, 5],
             [5, 5, 5]]
        ], dtype=np.int8)
        
        # Initialize with all unknown tiles (will be discovered by process)
        self.cube_colors = np.full((6, 3, 3), -1, dtype=np.int8)
    
    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Optional[Tuple[float, float, float]]]:
        # Find all unknown tiles (where value is -1)
        unknown_indices = np.where(self.cube_colors == -1)
        
        if len(unknown_indices[0]) > 0:
            # Pick the first unknown tile (discover in order: face, row, col)
            face = unknown_indices[0][0]
            row = unknown_indices[1][0]
            col = unknown_indices[2][0]
            
            # Reveal the color from the hardcoded cube
            self.cube_colors[face, row, col] = self.hardcoded_cube[face, row, col]
        
        return frame, self.cube_colors.copy(), None

