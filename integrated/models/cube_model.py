from PySide6.QtCore import QObject, Signal
import numpy as np
from typing import Tuple


class CubeModel(QObject):
    # Signal emitted when cube rotation changes
    rotation_changed = Signal(float, float, float)  # x, y, z rotation angles
    
    # Signal emitted when cube state changes (colors, etc.)
    cube_state_changed = Signal()
    
    def __init__(self):
        super().__init__()
        # Rotation angles in degrees
        self._rotation_x = 0.0
        self._rotation_y = 0.0
        self._rotation_z = 0.0
        
        # Cube colors: 6 faces, each with 9 squares (3x3)
        # Colors: 0=white, 1=yellow, 2=red, 3=orange, 4=green, 5=blue
        # Unknown color is -1
        self._colors = self._initialize_cube()
    
    def _initialize_cube(self) -> np.ndarray:
        # Shape: (6 faces, 3 rows, 3 cols)
        # Use int8 to support -1 for unknown colors
        colors = np.full((6, 3, 3), -1, dtype=np.int8)
        return colors
    
    @property
    def rotation(self) -> Tuple[float, float, float]:
        return (self._rotation_x, self._rotation_y, self._rotation_z)
    
    def set_rotation(self, x: float = None, y: float = None, z: float = None):
        changed = False
        if x is not None and self._rotation_x != x:
            self._rotation_x = x
            changed = True
        if y is not None and self._rotation_y != y:
            self._rotation_y = y
            changed = True
        if z is not None and self._rotation_z != z:
            self._rotation_z = z
            changed = True
        
        if changed:
            self.rotation_changed.emit(self._rotation_x, self._rotation_y, self._rotation_z)
    
    def rotate(self, delta_x: float = 0, delta_y: float = 0, delta_z: float = 0):
        self.set_rotation(
            self._rotation_x + delta_x,
            self._rotation_y + delta_y,
            self._rotation_z + delta_z
        )
    
    @property
    def colors(self) -> np.ndarray:
        return self._colors
    
    def set_colors(self, colors: np.ndarray):
        self._colors = colors
        self.cube_state_changed.emit()

