from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt, QTimer, QPoint
from PySide6.QtGui import QMouseEvent
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from OpenGL import GL
import numpy as np
import math


class CubeView(QWidget):
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
        self._rotation_x = 0.0
        self._rotation_y = 0.0
        self._rotation_z = 0.0
        
        # Disable auto-rotation - user will control via mouse drag
        self._auto_rotate = False
    
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
    
    def set_rotation(self, x: float, y: float, z: float):
        self._rotation_x = x
        self._rotation_y = y
        self._rotation_z = z
        self.gl_widget.set_rotation(x, y, z)
    
    def rotate_by_delta(self, delta_x: float, delta_y: float):
        """Rotate cube by delta angles (called from mouse drag)."""
        self._rotation_y += delta_x
        self._rotation_x += delta_y
        
        # Keep angles in reasonable range
        self._rotation_x = max(-90, min(90, self._rotation_x))
        
        self.gl_widget.set_rotation(self._rotation_x, self._rotation_y, self._rotation_z)
    
    def stop_animation(self):
        """Stop animation (kept for compatibility, but auto-rotation is disabled)."""
        pass


class RubiksCubeGLWidget(QOpenGLWidget):
    
    # Color palette for Rubik's cube faces
    COLORS = [
        (1.0, 1.0, 1.0),  # White
        (1.0, 1.0, 0.0),  # Yellow
        (1.0, 0.0, 0.0),  # Red
        (1.0, 0.5, 0.0),  # Orange
        (0.0, 1.0, 0.0),  # Green
        (0.0, 0.0, 1.0),  # Blue
    ]
    
    def __init__(self, cube_view: CubeView, parent=None):
        super().__init__(parent)
        self._cube_view = cube_view
        self._rotation_x = 0.0
        self._rotation_y = 0.0
        self._rotation_z = 0.0
        self._last_mouse_pos = None
        self._is_dragging = False
    
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
        # Map grid positions to face colors
        # Standard Rubik's cube: White=top, Yellow=bottom, Red=right, Orange=left, Blue=front, Green=back
        face_colors = {}
        
        # Front face (z = 2, facing camera) - Blue
        if grid_z == 2:
            face_colors['front'] = 4  # Blue
        # Back face (z = 0) - Green
        if grid_z == 0:
            face_colors['back'] = 5  # Green
        # Right face (x = 2) - Red
        if grid_x == 2:
            face_colors['right'] = 2  # Red
        # Left face (x = 0) - Orange
        if grid_x == 0:
            face_colors['left'] = 3  # Orange
        # Top face (y = 2) - White
        if grid_y == 2:
            face_colors['top'] = 0  # White
        # Bottom face (y = 0) - Yellow
        if grid_y == 0:
            face_colors['bottom'] = 1  # Yellow
        
        # Draw all 6 faces
        GL.glBegin(GL.GL_QUADS)
        
        # Front face
        if 'front' in face_colors:
            GL.glColor3f(*self.COLORS[face_colors['front']])
        else:
            GL.glColor3f(0.2, 0.2, 0.2)  # Dark gray for inner faces
        self._draw_face(x, y, z, size, 'front')
        
        # Back face
        if 'back' in face_colors:
            GL.glColor3f(*self.COLORS[face_colors['back']])
        else:
            GL.glColor3f(0.2, 0.2, 0.2)
        self._draw_face(x, y, z, size, 'back')
        
        # Right face
        if 'right' in face_colors:
            GL.glColor3f(*self.COLORS[face_colors['right']])
        else:
            GL.glColor3f(0.2, 0.2, 0.2)
        self._draw_face(x, y, z, size, 'right')
        
        # Left face
        if 'left' in face_colors:
            GL.glColor3f(*self.COLORS[face_colors['left']])
        else:
            GL.glColor3f(0.2, 0.2, 0.2)
        self._draw_face(x, y, z, size, 'left')
        
        # Top face
        if 'top' in face_colors:
            GL.glColor3f(*self.COLORS[face_colors['top']])
        else:
            GL.glColor3f(0.2, 0.2, 0.2)
        self._draw_face(x, y, z, size, 'top')
        
        # Bottom face
        if 'bottom' in face_colors:
            GL.glColor3f(*self.COLORS[face_colors['bottom']])
        else:
            GL.glColor3f(0.2, 0.2, 0.2)
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
    
    def set_rotation(self, x: float, y: float, z: float):
        self._rotation_x = x
        self._rotation_y = y
        self._rotation_z = z
        self.update()
    
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

