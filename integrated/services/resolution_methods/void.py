from typing import List
import numpy as np
from .base import ResolutionMethod


class VoidResolutionMethod(ResolutionMethod):
    """Resolution method that returns no moves (placeholder)."""
    
    def solve(self, cube_colors: np.ndarray) -> List[str]:
        return []

