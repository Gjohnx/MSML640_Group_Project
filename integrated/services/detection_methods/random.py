import numpy as np
import random
from typing import Tuple, Optional
from .base import DetectionMethod
from services.cube_rotations import CubeRotations


class RandomDetectionMethod(DetectionMethod):
    """Detection method that starts with a solved cube and applies random rotations."""
    
    # Standard Rubik's cube moves
    MOVES = ["R", "R'", "R2", "L", "L'", "L2", "U", "U'", "U2", 
             "D", "D'", "D2", "F", "F'", "F2", "B", "B'", "B2"]
    
    def __init__(self, num_scramble_moves: int = 20):
        """
        Initialize with a solved cube and scramble it.
        
        Args:
            num_scramble_moves: Number of random moves to apply to scramble the cube
        """
        self.num_scramble_moves = num_scramble_moves
        
        # Start with a solved cube
        # Face mapping: 0=U, 1=R, 2=F, 3=D, 4=L, 5=B
        self.scrambled_cube = np.array([
            # Face 0 (Up/White)
            [['U', 'U', 'U'],
             ['U', 'U', 'U'],
             ['U', 'U', 'U']],
            # Face 1 (Right/Blue)
            [['R', 'R', 'R'],
             ['R', 'R', 'R'],
             ['R', 'R', 'R']],
            # Face 2 (Front/Red)
            [['F', 'F', 'F'],
             ['F', 'F', 'F'],
             ['F', 'F', 'F']],
            # Face 3 (Down/Yellow)
            [['D', 'D', 'D'],
             ['D', 'D', 'D'],
             ['D', 'D', 'D']],
            # Face 4 (Left/Green)
            [['L', 'L', 'L'],
             ['L', 'L', 'L'],
             ['L', 'L', 'L']],
            # Face 5 (Back/Orange)
            [['B', 'B', 'B'],
             ['B', 'B', 'B'],
             ['B', 'B', 'B']]
        ], dtype=str)
        
        # Apply random scramble
        self._scramble_cube()
        
        # For progressive reveal (optional)
        self.cube_colors = self.scrambled_cube.copy()
        
        print(f"Random cube scrambled with {num_scramble_moves} moves")
    
    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Optional[Tuple[float, float, float]]]:
        return frame, self.cube_colors.copy(), None
    
    def _scramble_cube(self):
        scramble_sequence = []
        for _ in range(self.num_scramble_moves):
            move = random.choice(self.MOVES)
            scramble_sequence.append(move)
            self._apply_move(move)
        
        print(f"Scramble sequence: {' '.join(scramble_sequence)}")
    
    def _apply_move(self, move: str):
        self.scrambled_cube = CubeRotations.apply_move(self.scrambled_cube, move)

    def reset(self):
        pass