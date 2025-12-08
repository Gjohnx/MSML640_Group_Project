from typing import Dict
from .detection_methods import DetectionMethod, VoidDetectionMethod, HardcodedDetectionMethod, RandomDetectionMethod, YOLODetectionMethod
from .detection_methods.colorgrid import ColorGridDetectionMethod


class DetectionService:
    
    @staticmethod
    def get_all_detection_methods() -> Dict[str, DetectionMethod]:
        return {
            "Random": RandomDetectionMethod(num_scramble_moves=20),
            "Hardcoded": HardcodedDetectionMethod(),
            "HSV (Grid Rectify)": ColorGridDetectionMethod(),
            "YOLO": YOLODetectionMethod(),
            "Void": VoidDetectionMethod()

        }
