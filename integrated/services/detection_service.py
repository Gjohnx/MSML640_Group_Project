from typing import Dict
from .detection_methods import DetectionMethod, VoidDetectionMethod, ColorGridDetectionMethod, HardcodedDetectionMethod


class DetectionService:
    
    @staticmethod
    def get_all_detection_methods() -> Dict[str, DetectionMethod]:
        return {
            "Hardcoded": HardcodedDetectionMethod(),
            "Void": VoidDetectionMethod(),
            "Color Grid": ColorGridDetectionMethod(),
        }
