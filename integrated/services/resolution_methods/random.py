from typing import List
import numpy as np
import random
from .base import ResolutionMethod


class RandomResolutionMethod(ResolutionMethod):
    """Resolution method that suggests a random next move."""
    
    # Standard Rubik's cube moves
    MOVES = ["R", "L", "U", "D", "F", "B", "R'", "L'", "U'", "D'", "F'", "B'"]
    
    def solve(self, cube_colors: np.ndarray) -> List[str]:
        """
        Return a single random move.
        Since this is called once and we want one move per step,
        we return a list with one random move.
        """
        return [random.choice(self.MOVES)]

