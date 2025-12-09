from abc import ABC, abstractmethod
import numpy as np

class VideoFilter(ABC):
    
    @abstractmethod
    def apply(self, img: np.ndarray) -> np.ndarray:
        pass

