from PySide6.QtCore import QObject, Signal
from typing import Dict, Optional
from services.detection_methods import DetectionMethod


class ConfigurationModel(QObject):
    
    def __init__(self, detection_methods: Dict[str, DetectionMethod]):
        super().__init__()
        self._detection_methods: Dict[str, DetectionMethod] = detection_methods
        self._current_detection_method: Optional[str] = None
    
    def get_detection_method(self, name: str) -> Optional[DetectionMethod]:
        return self._detection_methods.get(name)
    
    @property
    def available_detection_methods(self) -> list[str]:
        return list(self._detection_methods.keys())
    
    @property
    def current_detection_method(self) -> Optional[str]:
        return self._current_detection_method
    
    @current_detection_method.setter
    def current_detection_method(self, value: Optional[str]):
        self._current_detection_method = value

