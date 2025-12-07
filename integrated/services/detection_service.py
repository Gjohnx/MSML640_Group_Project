# integrated/services/detection_service.py
from typing import Dict

from .detection_methods import (
    DetectionMethod,
    VoidDetectionMethod,
    HardcodedDetectionMethod,
    RandomDetectionMethod,
)

# Use the detect_adapter (wraps services/vision/color_grid_detector.py)
from .detection_methods.detect_adapter import DetectAdapterDetectionMethod
from .detection_methods.qbr_simple import QbrDetectionMethod

class DetectionService:
    @staticmethod
    def get_all_detection_methods() -> Dict[str, DetectionMethod]:
        """
        Return the detection methods available to the UI.
        Keys are the names shown in the dropdown.
        """
        return {
            "Random": RandomDetectionMethod(num_scramble_moves=20),
            "Hardcoded": HardcodedDetectionMethod(),
            "HSV (Grid Rectify)": DetectAdapterDetectionMethod(),  # <-- use adapter
            "Void": VoidDetectionMethod(),
            "QBR (Simple)": QbrDetectionMethod(), 
        }
