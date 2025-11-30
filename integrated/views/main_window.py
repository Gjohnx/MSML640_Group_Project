from PySide6.QtWidgets import QMainWindow, QWidget, QGridLayout
from PySide6.QtCore import Qt, QEvent
from .webcam_view import WebcamView
from .processed_view import ProcessedView
from .controls_view import ControlsView
from .cube_view import CubeView
from models.configuration_model import ConfigurationModel


class MainWindow(QMainWindow):

    def __init__(self, configuration_model: ConfigurationModel, parent=None):
        super().__init__(parent)
        self._cleanup_callback = None
        self._configuration_model = configuration_model
        self._init_ui()
    
    def set_cleanup_callback(self, callback):
        self._cleanup_callback = callback
    
    def _init_ui(self):
        self.setWindowTitle("Rubik's Cube Solver")
        self.setMinimumSize(1000, 700)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QGridLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)
        
        self.webcam_view = WebcamView()
        self.processed_view = ProcessedView()
        self.controls_view = ControlsView(self._configuration_model)
        self.cube_view = CubeView()
        
        # Add to grid 2x2 layout
        layout.addWidget(self.webcam_view, 0, 0)
        layout.addWidget(self.processed_view, 0, 1)
        layout.addWidget(self.controls_view, 1, 0)
        layout.addWidget(self.cube_view, 1, 1)
        
        # Set equal column and row stretch
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)
        layout.setRowStretch(0, 1)
        layout.setRowStretch(1, 1)
        
        central_widget.setLayout(layout)
    
    def get_webcam_view(self) -> WebcamView:
        return self.webcam_view
    
    def get_processed_view(self) -> ProcessedView:
        return self.processed_view
    
    def get_controls_view(self) -> ControlsView:
        return self.controls_view
    
    def get_cube_view(self) -> CubeView:
        return self.cube_view
    
    def closeEvent(self, event):
        # Stop cube animation timer
        self.cube_view.stop_animation()
        
        # Call cleanup callback if set
        if self._cleanup_callback:
            self._cleanup_callback()
        
        # Accept the close event
        event.accept()

