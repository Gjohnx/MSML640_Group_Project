from PySide6.QtCore import QObject, Signal
from enum import Enum


class AppState(Enum):
    """Application states for the Rubik's cube solver."""
    IDLE = "Idle"
    WAITING_FOR_DETECTION = "Waiting for detection"
    DETECTING = "Detecting"
    DETECTED = "Detected"
    RESOLVING = "Resolving"
    SOLVED = "Solved"


class StateModel(QObject):
    """Model that tracks the current application state."""
    
    # Signal emitted when state changes
    state_changed = Signal(AppState)
    
    def __init__(self):
        super().__init__()
        self._state = AppState.IDLE
    
    @property
    def state(self) -> AppState:
        """Get the current application state."""
        return self._state
    
    @state.setter
    def state(self, value: AppState):
        """Set the application state and emit signal if changed."""
        if self._state != value:
            self._state = value
            self.state_changed.emit(self._state)

