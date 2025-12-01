from typing import List
from abc import ABC, abstractmethod
import numpy as np


class ResolutionMethod(ABC):
    """
    Base class for Rubik's cube resolution methods.
    Resolution methods take a completed cube state and return a sequence of moves.
    """
    
    @abstractmethod
    def solve(self, cube_colors: np.ndarray) -> List[str]:
        """
        Solve the cube and return a list of moves.
        
        Args:
            cube_colors: A (6, 3, 3) numpy array representing the cube state.
                        Colors: 0=white, 1=yellow, 2=red, 3=orange, 4=green, 5=blue
        
        Returns:
            List of move strings (e.g., ["R", "U", "R'", "U'"])
        """
        pass

