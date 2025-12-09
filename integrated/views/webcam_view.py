from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
import cv2
import numpy as np


# The Webcam view is the widget that displays the raw webcam feed
class WebcamView(QWidget):
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
    
    def _init_ui(self):
        # Initialize the UI components
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Title label
        title = QLabel("Webcam Feed")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-weight: bold; font-size: 14px; padding: 5px;")
        
        # Video display label
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(320, 240)
        self.video_label.setStyleSheet("background-color: black; border: 1px solid gray;")
        self.video_label.setText("No webcam feed")
        
        layout.addWidget(title)
        layout.addWidget(self.video_label, stretch=1)
        
        self.setLayout(layout)
    
    def display_frame(self, frame: np.ndarray):
        # Display the Webcam
        if frame is None or frame.size == 0:
            return
        
        # Add two-boxes to the Webcam screen, we set the outer ratio is 0.75 and the inner ratio is 0.5
        # These two boxes are used for user to manually align the Rubik Cube
        h, w, _ = frame.shape
        S = min(h, w)
        outer_len = int(S * 0.5)
        inner_len = int(S * 0.25) 
        cx, cy = w // 2, h // 2
        ox1 = cx - outer_len // 2
        oy1 = cy - outer_len // 2
        ox2 = cx + outer_len // 2
        oy2 = cy + outer_len // 2
        ix1 = cx - inner_len // 2
        iy1 = cy - inner_len // 2
        ix2 = cx + inner_len // 2
        iy2 = cy + inner_len // 2

        # Display the boxes on screen
        cv2.rectangle(frame, (ox1, oy1), (ox2, oy2), (0, 255, 0), 2)   # green outer
        cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), (255, 0, 0), 2)   # blue inner

        # The frame sent by the controller is in BGR format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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

