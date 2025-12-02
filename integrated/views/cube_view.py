from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt, QTimer, QPoint
from PySide6.QtGui import QMouseEvent
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from OpenGL import GL
import numpy as np
import math
from typing import Tuple


class CubeView(QWidget):
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    @property
    def rotation(self) -> Tuple[float, float, float]:
        return self.gl_widget.rotation
    
    @rotation.setter
    def rotation(self, value: Tuple[float, float, float]):
        self.gl_widget.rotation = value

    @property
    def colors(self) -> np.ndarray:
        return self.gl_widget.colors
    
    @colors.setter
    def colors(self, colors: np.ndarray):
        self.gl_widget.colors = colors
    
    def _init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Title label
        title = QLabel("Detected Rubik's Cube")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-weight: bold; font-size: 14px; padding: 5px;")
        
        # OpenGL widget
        self.gl_widget = RubiksCubeGLWidget(self, self)
        self.gl_widget.setMinimumSize(320, 320)
        
        # Instruction label
        instruction_label = QLabel("Drag the mouse to rotate the cube")
        instruction_label.setAlignment(Qt.AlignCenter)
        instruction_label.setStyleSheet("font-size: 12px; padding: 5px; color: gray;")
        
        layout.addWidget(title)
        layout.addWidget(self.gl_widget, stretch=1)
        layout.addWidget(instruction_label)
        
        self.setLayout(layout)
    
    # Function to calculate the rotation angle based on the mouse drag
    def rotate_by_delta(self, delta_x: float, delta_y: float):
        x, y, z = self.gl_widget.rotation
        x += delta_y
        y += delta_x
        # Keep angles in reasonable range
        x = max(-180, min(180, x))
        y = max(-180, min(180, y))
        self.gl_widget.rotation = (x, y, z)

# This is the class that really manages the OpenGL rendering of the cube
class RubiksCubeGLWidget(QOpenGLWidget):
    
    # Color palette for Rubik's cube faces - maps character codes directly to RGB values
    # Based on hardcoded.py convention: U=white, R=blue, F=red, D=yellow, L=green, B=orange
    COLORS = {
        'U': (1.0, 1.0, 1.0),  # White
        'D': (1.0, 1.0, 0.0),  # Yellow
        'F': (1.0, 0.0, 0.0),  # Red
        'B': (1.0, 0.5, 0.0),  # Orange
        'R': (0.0, 0.0, 1.0),  # Blue
        'L': (0.0, 1.0, 0.0),  # Green
    }
    
    # Grey color for unknown tiles
    UNKNOWN_COLOR = (0.5, 0.5, 0.5)  # Grey
    
    def __init__(self, cube_view: CubeView, parent=None):
        super().__init__(parent)
        self._cube_view = cube_view
        self._colors = None
        self._rotation_x = 0.0
        self._rotation_y = 0.0
        self._rotation_z = 0.0
        self._last_mouse_pos = None
        self._is_dragging = False
    
    @property
    def colors(self) -> np.ndarray:
        return self._colors
    
    @colors.setter
    def colors(self, colors: np.ndarray):
        # Store colors as-is (can be strings or integers)
        # We'll handle conversion when rendering
        if colors is not None:
            self._colors = colors.copy()
        else:
            self._colors = colors
        self.update()

    @property
    def rotation(self) -> Tuple[float, float, float]:
        return (self._rotation_x, self._rotation_y, self._rotation_z)
    
    @rotation.setter
    def rotation(self, value: Tuple[float, float, float]):
        self._rotation_x, self._rotation_y, self._rotation_z = value
        self.update()

    def initializeGL(self):
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glClearColor(0.1, 0.1, 0.1, 1.0)
        GL.glEnable(GL.GL_CULL_FACE)
        GL.glCullFace(GL.GL_BACK)
    
    def resizeGL(self, width: int, height: int):
        GL.glViewport(0, 0, width, height)
    
    def paintGL(self):
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        
        # Perspective projection using glFrustum
        aspect = self.width() / self.height() if self.height() > 0 else 1.0
        fov = 45.0
        near = 0.1
        far = 100.0
        
        # Calculate frustum parameters
        fov_rad = math.radians(fov)
        f = 1.0 / math.tan(fov_rad / 2.0)
        top = near / f
        bottom = -top
        right = top * aspect
        left = -right
        
        # Use glFrustum for perspective projection
        GL.glFrustum(left, right, bottom, top, near, far)
        
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()
        
        # Camera position
        GL.glTranslatef(0.0, 0.0, -5.0)
        
        # Apply rotations
        GL.glRotatef(self._rotation_x, 1.0, 0.0, 0.0)
        GL.glRotatef(self._rotation_y, 0.0, 1.0, 0.0)
        GL.glRotatef(self._rotation_z, 0.0, 0.0, 1.0)
        
        # Draw the cube
        self._draw_cube()
    
    def _draw_cube(self):
        cube_size = 1.5
        spacing = 0.05
        cubelet_size = (cube_size - 2 * spacing) / 3
        
        # Draw 27 cubelets (3x3x3)
        for x in range(3):
            for y in range(3):
                for z in range(3):
                    pos_x = (x - 1) * (cubelet_size + spacing)
                    pos_y = (y - 1) * (cubelet_size + spacing)
                    pos_z = (z - 1) * (cubelet_size + spacing)
                    
                    self._draw_cubelet(pos_x, pos_y, pos_z, cubelet_size / 2, x, y, z)
    
    def _draw_cubelet(self, x: float, y: float, z: float, size: float, 
                     grid_x: int, grid_y: int, grid_z: int):
        # Map grid positions to face indices and tile positions
        # Using hardcoded.py convention: [U, R, F, D, L, B] (faces 0-5)
        # Face 0: U (White/Up), Face 1: R (Blue/Right), Face 2: F (Red/Front)
        # Face 3: D (Yellow/Down), Face 4: L (Green/Left), Face 5: B (Orange/Back)
        face_tiles = {}
        
        # Top face (y = 2): face 0 (U)
        # Looking down from above: back (grid_z=0) is top row (row=0), front (grid_z=2) is bottom row (row=2)
        # Left (grid_x=0) is left column (col=0), right (grid_x=2) is right column (col=2)
        if grid_y == 2:
            row = grid_z  # back=0 -> row 0, front=2 -> row 2
            col = grid_x  # left=0 -> col 0, right=2 -> col 2
            face_tiles['top'] = (0, row, col)
        
        # Right face (x = 2): face 1 (R)
        if grid_x == 2:
            row = 2 - grid_y
            col = 2 - grid_z
            face_tiles['right'] = (1, row, col)
        
        # Front face (z = 2): face 2 (F)
        # Looking at front: top (grid_y=2) is top row (row=0), bottom (grid_y=0) is bottom row (row=2)
        # Left (grid_x=0) is left column (col=0), right (grid_x=2) is right column (col=2)
        if grid_z == 2:
            row = 2 - grid_y  # top=2 -> row 0, bottom=0 -> row 2
            col = grid_x  # left=0 -> col 0, right=2 -> col 2
            face_tiles['front'] = (2, row, col)
        
        # Bottom face (y = 0): face 3 (D)
        # Looking up from below: front (grid_z=2) is top row (row=0), back (grid_z=0) is bottom row (row=2)
        # Right (grid_x=2) is left column (col=0), left (grid_x=0) is right column (col=2) - mirrored horizontally
        if grid_y == 0:
            row = 2 - grid_z  # front=2 -> row 0, back=0 -> row 2
            col = grid_x  # right=2 -> col 0, left=0 -> col 2
            face_tiles['bottom'] = (3, row, col)
        
        # Left face (x = 0): face 4 (L)
        if grid_x == 0:
            row = 2 - grid_y
            col = grid_z
            face_tiles['left'] = (4, row, col)
        
        # Back face (z = 0): face 5 (B)
        # Looking at back: top (grid_y=2) is top row (row=0), bottom (grid_y=0) is bottom row (row=2)
        # Right (grid_x=2) is left column (col=0), left (grid_x=0) is right column (col=2) - mirrored
        if grid_z == 0:
            row = 2 - grid_y  # top=2 -> row 0, bottom=0 -> row 2
            col = 2 - grid_x  # right=2 -> col 0, left=0 -> col 2
            face_tiles['back'] = (5, row, col)
        
        # Draw all 6 faces
        GL.glBegin(GL.GL_QUADS)
        
        # Helper function to get color for a face
        def get_face_color(face_key):
            if face_key not in face_tiles:
                return (0.2, 0.2, 0.2)  # Dark gray for inner faces
            
            if self._colors is None:
                return (0.2, 0.2, 0.2)  # Dark gray if no colors set
            
            face_idx, row, col = face_tiles[face_key]
            color_val = self._colors[face_idx, row, col]
            
            # Handle character codes directly (from hardcoded.py: 'U', 'R', 'F', 'D', 'L', 'B', or '?')
            if isinstance(color_val, (str, bytes)):
                char_val = str(color_val) if not isinstance(color_val, bytes) else color_val.decode('utf-8')
                if char_val == '?':
                    return self.UNKNOWN_COLOR  # Grey for unknown
                elif char_val in self.COLORS:
                    return self.COLORS[char_val]
                else:
                    return self.UNKNOWN_COLOR  # Grey for invalid character
            elif isinstance(color_val, np.generic):
                # Handle numpy scalar (could be integer or string)
                if color_val.dtype.kind == 'U' or color_val.dtype == object:
                    char_val = str(color_val)
                    if char_val == '?':
                        return self.UNKNOWN_COLOR
                    elif char_val in self.COLORS:
                        return self.COLORS[char_val]
                    else:
                        return self.UNKNOWN_COLOR
                else:
                    # Integer value (legacy support, but shouldn't happen with hardcoded.py)
                    return self.UNKNOWN_COLOR
            else:
                # Fallback for other types
                return self.UNKNOWN_COLOR
        
        # Front face
        GL.glColor3f(*get_face_color('front'))
        self._draw_face(x, y, z, size, 'front')
        
        # Back face
        GL.glColor3f(*get_face_color('back'))
        self._draw_face(x, y, z, size, 'back')
        
        # Right face
        GL.glColor3f(*get_face_color('right'))
        self._draw_face(x, y, z, size, 'right')
        
        # Left face
        GL.glColor3f(*get_face_color('left'))
        self._draw_face(x, y, z, size, 'left')
        
        # Top face
        GL.glColor3f(*get_face_color('top'))
        self._draw_face(x, y, z, size, 'top')
        
        # Bottom face
        GL.glColor3f(*get_face_color('bottom'))
        self._draw_face(x, y, z, size, 'bottom')
        
        GL.glEnd()
    
    def _draw_face(self, x: float, y: float, z: float, size: float, face: str):
        if face == 'front':
            GL.glVertex3f(x - size, y - size, z + size)
            GL.glVertex3f(x + size, y - size, z + size)
            GL.glVertex3f(x + size, y + size, z + size)
            GL.glVertex3f(x - size, y + size, z + size)
        elif face == 'back':
            GL.glVertex3f(x + size, y - size, z - size)
            GL.glVertex3f(x - size, y - size, z - size)
            GL.glVertex3f(x - size, y + size, z - size)
            GL.glVertex3f(x + size, y + size, z - size)
        elif face == 'right':
            GL.glVertex3f(x + size, y - size, z + size)
            GL.glVertex3f(x + size, y - size, z - size)
            GL.glVertex3f(x + size, y + size, z - size)
            GL.glVertex3f(x + size, y + size, z + size)
        elif face == 'left':
            GL.glVertex3f(x - size, y - size, z - size)
            GL.glVertex3f(x - size, y - size, z + size)
            GL.glVertex3f(x - size, y + size, z + size)
            GL.glVertex3f(x - size, y + size, z - size)
        elif face == 'top':
            GL.glVertex3f(x - size, y + size, z + size)
            GL.glVertex3f(x + size, y + size, z + size)
            GL.glVertex3f(x + size, y + size, z - size)
            GL.glVertex3f(x - size, y + size, z - size)
        elif face == 'bottom':
            GL.glVertex3f(x - size, y - size, z - size)
            GL.glVertex3f(x + size, y - size, z - size)
            GL.glVertex3f(x + size, y - size, z + size)
            GL.glVertex3f(x - size, y - size, z + size)
    
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self._is_dragging = True
            self._last_mouse_pos = event.position().toPoint()
            self.setCursor(Qt.ClosedHandCursor)
    
    def mouseMoveEvent(self, event: QMouseEvent):
        if self._is_dragging and self._last_mouse_pos is not None:
            current_pos = event.position().toPoint()
            delta_x = current_pos.x() - self._last_mouse_pos.x()
            delta_y = current_pos.y() - self._last_mouse_pos.y()
            
            # Convert mouse movement to rotation (sensitivity factor)
            rotation_sensitivity = 0.5
            self._cube_view.rotate_by_delta(
                delta_x * rotation_sensitivity,
                delta_y * rotation_sensitivity
            )
            
            self._last_mouse_pos = current_pos
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self._is_dragging = False
            self._last_mouse_pos = None
            self.setCursor(Qt.ArrowCursor)
    
    def cleanup(self):
        self.makeCurrent()
        self.doneCurrent()

