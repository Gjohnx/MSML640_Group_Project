from typing import Optional
from abc import ABC, abstractmethod
import numpy as np


class ResolutionMethod(ABC):
    
    @abstractmethod
    def solve(self, cube_colors: np.ndarray) -> Optional[str]:
        pass

    @abstractmethod
    def undo(self):
        pass