import numpy as np
from models.webcam_model import WebcamModel
from models.configuration_model import ConfigurationModel
from views.processed_view import ProcessedView


class ProcessingController:
    
    def __init__(self, webcam_model: WebcamModel, configuration_model: ConfigurationModel, 
                 view: ProcessedView):
        self.webcam_model = webcam_model
        self.configuration_model = configuration_model
        self.view = view
        
        # Connect signals
        self.webcam_model.frame_captured.connect(self._process_frame)
    
    def _process_frame(self, frame: np.ndarray):
        """Process a frame with the selected algorithm."""
        algorithm_name = self.configuration_model.current_algorithm
        if not algorithm_name:
            # No algorithm selected, show original
            self.view.display_frame(frame)
            return
        
        algorithm_func = self.configuration_model.get_algorithm(algorithm_name)
        if algorithm_func:
            try:
                processed_frame = algorithm_func(frame)
                self.view.display_frame(processed_frame)
            except Exception as e:
                # On error, show original frame
                print(f"Error processing frame: {e}")
                self.view.display_frame(frame)
        else:
            self.view.display_frame(frame)

