from PySide6.QtCore import QObject, Signal
import numpy as np


# The Webcam model contains the state of the webcam, as the captured frame
# In reality it doesn't persist the frame and only emits a signal
# The Controller cannot emit a signal, only the model can
# The signal contains the frame without needing to persist it
class WebcamModel(QObject):
    
    # When a Frame is captured, a Signal is emitted
    # Captured Frame means that it has been obtained from the webcam
    frame_captured = Signal(np.ndarray)
    
    # This function is called by the WebcamController when a new frame is captured
    def update_frame(self, frame: np.ndarray):
        self.frame_captured.emit(frame)

