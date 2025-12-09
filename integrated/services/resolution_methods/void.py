import numpy as np
from typing import Optional
from .base import ResolutionMethod


class VoidResolutionMethod(ResolutionMethod):
    """Resolution method that does nothing (returns no moves)."""
    
    def __init__(self):
        pass
    
    def solve(self, cube_colors: np.ndarray) -> Optional[str]:
        """Return no move."""
        return None

    def undo(self):
        """No operation."""
        pass