from typing import Tuple, Optional
from abc import ABC, abstractmethod
import numpy as np


class DetectionMethod(ABC):
    
    @abstractmethod
    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Optional[Tuple[float, float, float]]]:
        # Process a frame and return processed frame, cube colors, and optional rotation.
        pass

