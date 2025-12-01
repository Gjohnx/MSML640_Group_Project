from PySide6.QtCore import QObject, Signal
from typing import Dict
from services.detection_methods import DetectionMethod
from services.resolution_methods import ResolutionMethod

# The Configuration model contains the state of the application configuration, as the detection method selected
class ConfigurationModel(QObject):
    
    def __init__(self, detection_methods: Dict[str, DetectionMethod], 
                 resolution_methods: Dict[str, ResolutionMethod]):
        super().__init__()
        self._detection_methods: Dict[str, DetectionMethod] = detection_methods
        self._resolution_methods: Dict[str, ResolutionMethod] = resolution_methods
        
        # The first available method is selected by default, it can be changed later
        available_detection = list(self._detection_methods.keys())
        available_resolution = list(self._resolution_methods.keys())
        
        self.current_detection_method: str = available_detection[0] if available_detection else ""
        self.current_resolution_method: str = available_resolution[0] if available_resolution else ""
    
    def get_detection_method(self, name: str) -> DetectionMethod | None:
        return self._detection_methods.get(name)
    
    @property
    def available_detection_methods(self) -> list[str]:
        return list(self._detection_methods.keys())
    
    def get_resolution_method(self, name: str) -> ResolutionMethod | None:
        return self._resolution_methods.get(name)
    
    @property
    def available_resolution_methods(self) -> list[str]:
        return list(self._resolution_methods.keys())

