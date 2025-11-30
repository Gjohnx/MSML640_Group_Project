from typing import Dict
from .detection_methods import DetectionMethod, VoidDetectionMethod, RandomDetectionMethod


class DetectionService:
    
    @staticmethod
    def get_all_detection_methods() -> Dict[str, DetectionMethod]:
        return {
            "Void": VoidDetectionMethod(),
            "Random": RandomDetectionMethod(),
        }
