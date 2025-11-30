from PySide6.QtCore import QObject, Signal
import numpy as np


class WebcamModel(QObject):
    
    # When a Frame is captured, a Signal is emitted
    # Captured Frame means that it has been obtained from the webcam
    frame_captured = Signal(np.ndarray)
    
    # This function is called by the WebcamController when a new frame is captured
    def update_frame(self, frame: np.ndarray):
        self.frame_captured.emit(frame)

