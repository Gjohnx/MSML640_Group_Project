from PySide6.QtWidgets import QMainWindow, QWidget, QGridLayout
from PySide6.QtCore import Qt, QEvent
from .webcam_view import WebcamView
from .processed_view import ProcessedView
from .controls_view import ControlsView
from .cube_view import CubeView

# The QT class that creates the application window
class MainWindow(QMainWindow):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cleanup_callback = None
        self._init_ui()
    
    # The function to call when the window is closed, defined by the AppController
    def set_cleanup_callback(self, callback):
        self._cleanup_callback = callback
    
    def _init_ui(self):
        # Create the window with grid layout containing the different sections (Webcam, control, 3D cube, etc)
        # Each of these sections is a view
        self.setWindowTitle("Rubik's Cube Solver")
        self.setMinimumSize(1000, 700)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QGridLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Create the views
        self._webcam_view = WebcamView()
        self._processed_view = ProcessedView()
        self._controls_view = ControlsView()
        self._cube_view = CubeView()
        
        # Add the views to the grid layout
        layout.addWidget(self._webcam_view, 0, 0)
        layout.addWidget(self._processed_view, 0, 1)
        layout.addWidget(self._controls_view, 1, 0)
        layout.addWidget(self._cube_view, 1, 1)
        
        # Set columns and rows for resizing
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)
        layout.setRowStretch(0, 1)
        layout.setRowStretch(1, 1)
        
        central_widget.setLayout(layout)
    
    def get_webcam_view(self) -> WebcamView:
        return self._webcam_view
    
    def get_processed_view(self) -> ProcessedView:
        return self._processed_view
    
    def get_controls_view(self) -> ControlsView:
        return self._controls_view
    
    def get_cube_view(self) -> CubeView:
        return self._cube_view
    
    # Redefine function from the QMainWindow class
    # When the window is closed, the callback is called to stop the application
    def closeEvent(self, event):
        # Call cleanup callback if set
        if self._cleanup_callback:
            self._cleanup_callback()
        
        # Accept the close event
        event.accept()

