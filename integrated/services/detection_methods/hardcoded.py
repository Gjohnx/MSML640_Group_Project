import numpy as np
from typing import Tuple, Optional
from .base import DetectionMethod


class HardcodedDetectionMethod(DetectionMethod):
    
    def __init__(self):
        # Hardcoded cube matrix (6 faces, 3 rows, 3 columns)
        # Colors: U=white, R=blue, F=red, D=yellow, L=green, B=orange
        self.hardcoded_cube = np.array([
            # Face 0 (White/Up)
            [['U', 'U', 'U'],
             ['U', 'U', 'U'],
             ['L', 'L', 'L']],
            # Face 1 (Blue/Right)
            [['U', 'R', 'R'],
             ['U', 'R', 'R'],
             ['U', 'R', 'R']],
            # Face 2 (Red/Front)
            [['F', 'F', 'F'],
             ['F', 'F', 'F'],
             ['F', 'F', 'F']],
            # Face 3 (Yellow/Down)
            [['R', 'R', 'R'],
             ['D', 'D', 'D'],
             ['D', 'D', 'D']],
            # Face 4 (Green/Left)
            [['L', 'L', 'D'],
             ['L', 'L', 'D'],
             ['L', 'L', 'D']],
            # Face 5 (Orange/Back)
            [['B', 'B', 'B'],
             ['B', 'B', 'B'],
             ['B', 'B', 'B']]
        ], dtype=str)
        
        # Initialize with all unknown tiles (will be discovered by process)
        self.cube_colors = np.full((6, 3, 3), '?', dtype=str)
    
    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Optional[Tuple[float, float, float]]]:
        # Find all unknown tiles (where value is -1)
        unknown_indices = np.where(self.cube_colors == '?')
        
        if len(unknown_indices[0]) > 0:
            # Pick the first unknown tile (discover in order: face, row, col)
            face = unknown_indices[0][0]
            row = unknown_indices[1][0]
            col = unknown_indices[2][0]
            
            # Reveal the color from the hardcoded cube
            self.cube_colors[face, row, col] = self.hardcoded_cube[face, row, col]
        
        return frame, self.cube_colors.copy(), None

