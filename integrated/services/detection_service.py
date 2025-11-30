from typing import Dict
from .detection_methods import DetectionMethod, VoidDetectionMethod, RandomDetectionMethod, ColorGridDetectionMethod


class DetectionService:
    
    @staticmethod
    def get_all_detection_methods() -> Dict[str, DetectionMethod]:
        return {
            "Random": RandomDetectionMethod(),
            "Color Grid": ColorGridDetectionMethod(),
            "Void": VoidDetectionMethod(),
        }
