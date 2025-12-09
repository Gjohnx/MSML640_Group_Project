from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
import cv2
import numpy as np

# This view displays the Webcam feed after it has been processed by the detection method
class ProcessedView(QWidget):
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
    
    def _init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Title label
        title = QLabel("Processed Feed")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-weight: bold; font-size: 14px; padding: 5px;")
        
        # Video display label
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(320, 240)
        self.video_label.setStyleSheet("background-color: black; border: 1px solid gray;")
        self.video_label.setText("No processed feed")
        
        layout.addWidget(title)
        layout.addWidget(self.video_label, stretch=1)
        
        self.setLayout(layout)
    
    def display_frame(self, frame: np.ndarray):
        if frame is None or frame.size == 0:
            return
        
        # The frame sent by the controller is in BGR format, so we need to convert it to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get the shape of the frame
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        
        # Create QImage from numpy array
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Scale to fit label while maintaining aspect ratio
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        self.video_label.setPixmap(scaled_pixmap)

