import numpy as np
from models.webcam_model import WebcamModel
from models.configuration_model import ConfigurationModel
from models.cube_model import CubeModel
from views.processed_view import ProcessedView


class ProcessingController:
    
    def __init__(self, webcam_model: WebcamModel, configuration_model: ConfigurationModel, 
                 view: ProcessedView, cube_model: CubeModel = None):
        self.webcam_model = webcam_model
        self.configuration_model = configuration_model
        self.view = view
        self.cube_model = cube_model
        
        # Connect signals
        self.webcam_model.frame_captured.connect(self._process_frame)
    
    def _process_frame(self, frame: np.ndarray):
        detection_method_name = self.configuration_model.current_detection_method
        if not detection_method_name:
            # No detection method selected, show original
            self.view.display_frame(frame)
            return
        
        detection_method = self.configuration_model.get_detection_method(detection_method_name)
        if detection_method:
            try:
                processed_frame, cube_colors = detection_method.process(frame)
                if self.cube_model is not None:
                        self.cube_model.set_colors(cube_colors)
                self.view.display_frame(processed_frame)
            except Exception as e:
                # On error, show original frame
                print(f"Error processing frame: {e}")
                self.view.display_frame(frame)
        else:
            self.view.display_frame(frame)

