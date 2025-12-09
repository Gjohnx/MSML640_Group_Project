from typing import Dict
from .detection_methods import DetectionMethod, VoidDetectionMethod, HardcodedDetectionMethod, RandomDetectionMethod
from .detection_methods.deepl import YOLODetectionMethod, YOLOCNNDetectionMethod
from .detection_methods.colorgrid import ColorGridDetectionMethod
from .detection_methods.qbr_simple import QbrDetectionMethod

class DetectionService:
    
    @staticmethod
    def get_all_detection_methods() -> Dict[str, DetectionMethod]:
        return {
            "YOLO+CNN": YOLOCNNDetectionMethod(),
            "QBR (Simple)": QbrDetectionMethod(), 
            "HSV (Grid Rectify)": ColorGridDetectionMethod(),
            "YOLO": YOLODetectionMethod(),
            "Random": RandomDetectionMethod(num_scramble_moves=20),
            "Hardcoded": HardcodedDetectionMethod(),
            "Void": VoidDetectionMethod(),

        }
