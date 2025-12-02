from typing import Dict
from .detection_methods import DetectionMethod, VoidDetectionMethod, HardcodedDetectionMethod, RandomDetectionMethod


class DetectionService:
    
    @staticmethod
    def get_all_detection_methods() -> Dict[str, DetectionMethod]:
        return {
            "Random": RandomDetectionMethod(num_scramble_moves=20),
            "Hardcoded": HardcodedDetectionMethod(),
            "Void": VoidDetectionMethod()
        }
