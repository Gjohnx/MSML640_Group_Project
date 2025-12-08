from .base import DetectionMethod
from .void import VoidDetectionMethod
from .hardcoded import HardcodedDetectionMethod
from .random import RandomDetectionMethod
from .yolo import YOLODetectionMethod

__all__ = ['DetectionMethod', 'VoidDetectionMethod', 'HardcodedDetectionMethod', 'RandomDetectionMethod', 'YOLODetectionMethod']

