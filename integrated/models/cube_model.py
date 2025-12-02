from PySide6.QtCore import QObject, Signal
import numpy as np
from typing import Tuple
from models.state_model import AppState, StateModel


# The Cube model contains the state of the cube, the known colors and the known rotation angles
class CubeModel(QObject):
    # When the rotation angles are changed by the controller, this signal is emitted to update the view
    rotation_changed = Signal(float, float, float)  # x, y, z rotation angles
    
    # When the cube colors are changed by the controller, this signal is emitted to update the view
    cube_state_changed = Signal()
    
    def __init__(self, state_model: StateModel):
        super().__init__()
        # Rotation angles in degrees
        # Start with a slight rotation of 15 degrees for a nicer view on startup
        self._rotation_x = 15.0
        self._rotation_y = 15.0
        self._rotation_z = 0.0
        
        # The cube is stored as a Numpy array with the following dimensions (6 faces, 3 rows, 3 columns)
        # The colors are stored as strings, where 'U'=white, 'D'=yellow, 'F'=red, 'B'=orange, 'R'=blue, 'L'=green, '?'=unknown
        self._colors = self._initialize_cube()
        
        self.state_model = state_model
    
    def _initialize_cube(self) -> np.ndarray:
        # Initialize the cube with all unknown colors
        # Use string '?' for unknown colors
        colors = np.full((6, 3, 3), '?', dtype=str)
        return colors
    
    @property
    def rotation(self) -> Tuple[float, float, float]:
        # Get the current rotation angles
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
    
    # Get the current cube colors
    @property
    def colors(self) -> np.ndarray:
        return self._colors

    # Set the cube colors
    @colors.setter
    def colors(self, colors: np.ndarray):
        self._colors = colors
        self.cube_state_changed.emit()
        
        # Check if cube is completed (no unknown tiles)
        if np.all(self._colors != '?') and (self.state_model.state == AppState.DETECTING):
            self.state_model.state = AppState.DETECTED
        
        # Check if cube is solved (all faces have uniform colors)
        # if self._is_solved() and self.state_model.state == AppState.RESOLVING:
        #     self.state_model.state = AppState.SOLVED
    
    def _is_solved(self) -> bool:
        # Check if there are any unknown tiles
        if np.any(self._colors == '?'):
            return False
        
        # Check if each face has uniform color
        for face_idx in range(6):
            face = self._colors[face_idx]
            # Get the color of the center tile
            center_color = face[1, 1]
            # Check if all tiles on this face have the same color
            if not np.all(face == center_color):
                return False
        
        return True

