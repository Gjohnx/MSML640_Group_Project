from models.webcam_model import WebcamModel
from models.configuration_model import ConfigurationModel
from models.cube_model import CubeModel
from views.main_window import MainWindow
from controllers.webcam_controller import WebcamController
from controllers.processing_controller import ProcessingController
from controllers.cube_controller import CubeController
from controllers.controls_controller import ControlsController
from services.algorithm_service import AlgorithmService


class AppController:
    
    def __init__(self):
        
        # Get all algorithms from service layer
        algorithms = AlgorithmService.get_all_algorithms()
        
        # Create models
        self.webcam_model = WebcamModel()
        self.configuration_model = ConfigurationModel(algorithms)
        self.cube_model = CubeModel()
        
        # Create main window (pass configuration model so views can access it)
        self.main_window = MainWindow(self.configuration_model)
        
        # Create controllers
        self.webcam_controller = WebcamController(
            self.webcam_model,
            self.main_window.get_webcam_view()
        )
        
        self.processing_controller = ProcessingController(
            self.webcam_model,
            self.configuration_model,
            self.main_window.get_processed_view()
        )
        
        self.cube_controller = CubeController(
            self.cube_model,
            self.main_window.get_cube_view()
        )
        
        self.controls_controller = ControlsController(
            self.configuration_model,
            self.main_window.get_controls_view()
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
        
        # Stop cube animation
        cube_view = self.main_window.get_cube_view()
        cube_view.stop_animation()

