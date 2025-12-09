from typing import Dict
from .detection_methods import DetectionMethod, VoidDetectionMethod, HardcodedDetectionMethod, RandomDetectionMethod
from .detection_methods.colorgrid import ColorGridDetectionMethod
from services.detection_methods.processed_comparison_detection_method import ProcessedComparisonDetectionMethod


class DetectionService:
    
    @staticmethod
    def get_all_detection_methods() -> Dict[str, DetectionMethod]:
        return {
            "Random": RandomDetectionMethod(num_scramble_moves=20),
            "Hardcoded": HardcodedDetectionMethod(),
            "HSV (Grid Rectify)": ColorGridDetectionMethod(),
            "Void": VoidDetectionMethod(),
            "Processed Comparison": ProcessedComparisonDetectionMethod()
        }
