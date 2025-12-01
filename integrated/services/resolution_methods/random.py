import numpy as np
import random
from .base import ResolutionMethod


class RandomResolutionMethod(ResolutionMethod):
    """Resolution method that suggests a random next move."""
    
    # Standard Rubik's cube moves
    MOVES = ["R", "R'", "L", "L'", "U", "U'", "D", "D'", "F", "F'", "B", "B'"]

    def __init__(self):
        self.next_move_idx = 0
    
    def solve(self, cube_colors: np.ndarray) -> str:
        """Return a single random move."""
        move = self.MOVES[self.next_move_idx]
        self.next_move_idx = (self.next_move_idx + 1) % len(self.MOVES)
        return move

