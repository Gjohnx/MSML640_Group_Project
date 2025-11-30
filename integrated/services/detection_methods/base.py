from typing import Tuple
from abc import ABC, abstractmethod
import numpy as np


class DetectionMethod(ABC):
    
    @abstractmethod
    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pass

