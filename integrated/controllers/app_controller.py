from models.webcam_model import WebcamModel
from models.configuration_model import ConfigurationModel
from models.cube_model import CubeModel
from models.state_model import StateModel
from views.main_window import MainWindow
from controllers.webcam_controller import WebcamController
from controllers.processing_controller import ProcessingController
from controllers.cube_controller import CubeController
from controllers.controls_controller import ControlsController
from services.detection_service import DetectionService
from services.resolution_service import ResolutionService


class AppController:
    
    def __init__(self):
        
        # Get all detection and resolution methods from service layer
        detection_methods = DetectionService.get_all_detection_methods()
        resolution_methods = ResolutionService.get_all_resolution_methods()
        
        # Step 1: Create all Views first
        self.main_window = MainWindow()
        
        # Step 2: Create all Models
        self.configuration_model = ConfigurationModel(detection_methods, resolution_methods)
        self.webcam_model = WebcamModel()
        self.state_model = StateModel()
        self.cube_model = CubeModel(self.state_model)
        
        # Step 3: Create all Controllers (they will connect signals)
        self.webcam_controller = WebcamController(
            self.webcam_model,
            self.main_window.get_webcam_view()
        )
        
        self.processing_controller = ProcessingController(
            self.webcam_model,
            self.configuration_model,
            self.main_window.get_processed_view(),
            self.cube_model,
            self.state_model,
            self.main_window.get_controls_view()
        )
        
        self.cube_controller = CubeController(
            self.cube_model,
            self.main_window.get_cube_view()
        )
        
        self.controls_controller = ControlsController(
            self.configuration_model,
            self.main_window.get_controls_view(),
            self.cube_model,
            self.state_model
        )
    
    def start(self):
        # Set cleanup callback for window close
        self.main_window.set_cleanup_callback(self.stop)
        self.main_window.show()
        # Start webcam capture
        self.webcam_controller.start_capture()
    
    def stop(self):
        # Stop webcam capture
        self.webcam_controller.stop_capture()

