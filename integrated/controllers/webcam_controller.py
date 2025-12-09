from PySide6.QtCore import QThread, Signal, QTimer
import cv2
import numpy as np
from models.webcam_model import WebcamModel
from views.webcam_view import WebcamView


class WebcamThread(QThread):
    
    frame_ready = Signal(np.ndarray)
    error_occurred = Signal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._running = False
        self._camera = None
    
    def run(self):
        self._camera = cv2.VideoCapture(0)

        # Force 720p resolution (the following 2 lines are for windows only. In macboook, it should has no influence)
        self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not self._camera.isOpened():
            self.error_occurred.emit("Failed to open webcam")
            return
        
        self._running = True
        
        while self._running and not self.isInterruptionRequested():
            ret, frame = self._camera.read()
            if ret:
                self.frame_ready.emit(frame)
            else:
                self.error_occurred.emit("Failed to read frame")
                break
            
            # Fix that for windows systems since the original one cause low FPS (likely 2 FPS) for windows (below)
            self.msleep(1)

            # Small delay to control frame rate, but check for stop more frequently
            # for _ in range(33):  # Break into 1ms chunks to be more responsive
                # if not self._running or self.isInterruptionRequested():
                    # break
                # self.msleep(1)
        
        # Cleanup camera
        if self._camera:
            self._camera.release()
            self._camera = None
    
    def stop(self):
        self._running = False
        self.requestInterruption()
        
        # Wait for thread to finish with timeout (should be quick now with 1ms sleeps)
        if self.isRunning():
            if not self.wait(2000):  # 2 second timeout
                # If thread doesn't stop, log but don't force terminate (avoids segfault)
                print("Warning: Webcam thread did not stop within timeout")
        
        # Camera is released in run() method after loop exits


class WebcamController:
    
    def __init__(self, model: WebcamModel, view: WebcamView):
        self.model = model
        self.view = view
        self._thread = None
        
        # Connect model signals to view
        self.model.frame_captured.connect(self.view.display_frame)
    
    def start_capture(self):
        if self._thread is None or not self._thread.isRunning():
            self._thread = WebcamThread()
            # Connect the frame_ready signal to the update_frame function
            # This function will update the frame in the model and emit the frame_captured signal
            self._thread.frame_ready.connect(self.model.update_frame)
            self._thread.error_occurred.connect(self._on_error)
            self._thread.start()
    
    def stop_capture(self):
        if self._thread:
            if self._thread.isRunning():
                self._thread.stop()
            # Disconnect signals to prevent any pending signals from causing issues
            try:
                self._thread.frame_ready.disconnect()
                self._thread.error_occurred.disconnect()
            except Exception as e:
                # Debug 
                print(f"Warning: Signal disconnection failed: {e}")
            self._thread = None
            # Update view to show webcam stopped
            self.view.video_label.clear()
            self.view.video_label.setText("Webcam stopped")
    
    def _on_error(self, error_message: str):
        self.view.video_label.clear()
        self.view.video_label.setText(f"Error: {error_message}")

