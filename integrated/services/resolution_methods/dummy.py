import numpy as np
import random
from .base import ResolutionMethod


class DummyResolutionMethod(ResolutionMethod):
    # Standard Rubik's cube moves including double moves
    MOVES = ["R", "R'", "R2", "R2", "L", "L'", "L2", "L2", "U", "U'", "U2", "U2", 
             "D", "D'", "D2", "D2", "F", "F'", "F2", "F2", "B", "B'", "B2", "B2"]

    def __init__(self):
        self.next_move_idx = 0
    
    def solve(self, cube_colors: np.ndarray) -> str:
        move = self.MOVES[self.next_move_idx]
        self.next_move_idx = (self.next_move_idx + 1) % len(self.MOVES)
        return move

    def undo(self):
        self.next_move_idx = (self.next_move_idx - 1) % len(self.MOVES)