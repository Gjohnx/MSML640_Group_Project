from PySide6.QtCore import QThread, Signal
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
        # 0 is usually the default webcam. You can try 1 or 2 if you have multiple cameras.
        self._camera = cv2.VideoCapture(0)
        
        # Optimize camera settings for speed (optional, might not work on all cameras)
        self._camera.set(cv2.CAP_PROP_FPS, 30)
        
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
            
            # --- REMOVED THE ARTIFICIAL 33ms DELAY LOOP ---
            # The camera hardware naturally limits the FPS (usually 30 or 60).
            # We only add a tiny sleep to yield execution and prevent CPU hogging
            # if the camera read returns instantly (unlikely, but safe).
            self.msleep(1)
        
        # Cleanup camera
        if self._camera:
            self._camera.release()
            self._camera = None
    
    def stop(self):
        self._running = False
        self.requestInterruption()
        
        # Wait for thread to finish with timeout
        if self.isRunning():
            if not self.wait(2000):  # 2 second timeout
                print("Warning: Webcam thread did not stop within timeout")


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
            # Connect signals
            self._thread.frame_ready.connect(self.model.update_frame)
            self._thread.error_occurred.connect(self._on_error)
            self._thread.start()
    
    def stop_capture(self):
        if self._thread:
            if self._thread.isRunning():
                self._thread.stop()
            # Disconnect signals
            try:
                self._thread.frame_ready.disconnect()
                self._thread.error_occurred.disconnect()
            except Exception as e:
                print(f"Warning: Signal disconnection failed: {e}")
            self._thread = None
            # Update view
            self.view.video_label.clear()
            self.view.video_label.setText("Webcam stopped")
    
    def _on_error(self, error_message: str):
        self.view.video_label.clear()
        self.view.video_label.setText(f"Error: {error_message}")