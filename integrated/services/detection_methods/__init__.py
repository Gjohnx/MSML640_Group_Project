from .base import DetectionMethod
from .void import VoidDetectionMethod
from .hardcoded import HardcodedDetectionMethod
from .random import RandomDetectionMethod
from .deepl.yolo import YOLODetectionMethod
from .deepl.yolo_cnn import YOLOCNNDetectionMethod

__all__ = ['DetectionMethod', 'VoidDetectionMethod', 'HardcodedDetectionMethod', 'RandomDetectionMethod', 'YOLODetectionMethod', 'YOLOCNNDetectionMethod']

