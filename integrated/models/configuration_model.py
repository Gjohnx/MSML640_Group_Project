from PySide6.QtCore import QObject, Signal
from typing import Dict, Callable, Optional


class ConfigurationModel(QObject):
    
    def __init__(self, algorithms: Dict[str, Callable]):
        super().__init__()
        self._algorithms: Dict[str, Callable] = algorithms
        self._current_algorithm: Optional[str] = None
    
    def get_algorithm(self, name: str) -> Optional[Callable]:
        return self._algorithms.get(name)
    
    @property
    def available_algorithms(self) -> list[str]:
        return list(self._algorithms.keys())
    
    @property
    def current_algorithm(self) -> Optional[str]:
        return self._current_algorithm
    
    @current_algorithm.setter
    def current_algorithm(self, value: Optional[str]):
        self._current_algorithm = value

