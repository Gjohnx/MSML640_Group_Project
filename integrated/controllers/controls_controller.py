from models.configuration_model import ConfigurationModel
from models.cube_model import CubeModel
from models.state_model import StateModel, AppState
from views.controls_view import ControlsView


class ControlsController:
    
    def __init__(self, configuration_model: ConfigurationModel, view: ControlsView, 
                 cube_model: CubeModel, state_model: StateModel):
        self.configuration_model = configuration_model
        self.view = view
        self.cube_model = cube_model
        self.state_model = state_model
        
        # Populate view with methods from configuration model
        self.view.set_detection_methods(self.configuration_model.available_detection_methods)
        self.view.set_resolution_methods(self.configuration_model.available_resolution_methods)
        
        # Connect view signals to state model
        self.view.start_detection_clicked.connect(self._on_start_detection_clicked)
        self.view.start_resolution_clicked.connect(self._on_start_resolution_clicked)
        self.view.reset_clicked.connect(self._on_reset_clicked)
        self.view.detect_clicked.connect(self._on_detect_clicked)
        # self.view.next_step_clicked.connect(self._on_next_step_clicked)
        
        # Connect model signals to view
        self.state_model.state_changed.connect(self._on_state_changed)
    
    def _on_state_changed(self, state: AppState):
        if state == AppState.DETECTING:
            self.view.disable_start_detection()
            self.view.disable_detection_method()
            self.view.enable_reset()
            self.view.enable_detect()
        elif state == AppState.DETECTED:
            self.view.enable_start_resolution()
            self.view.enable_resolution_method()
            self.view.disable_reset()
            self.view.disable_detect()
        elif state == AppState.RESOLVING:
            self.view.disable_resolution_method()
            self.view.disable_start_resolution()
            self.view.enable_next_step()
            self.view.enable_prev_step()
        elif state == AppState.SOLVED:
            self.view.disable_next_step()
            self.view.disable_prev_step()

    def _on_reset_clicked(self):
        # self.cube_model.reset()
        pass

    def _on_detect_clicked(self):
        # self.state_model.state = AppState.DETECTING
        pass

    def _on_start_detection_clicked(self, detection_method: str):
        self.configuration_model.current_detection_method = detection_method
        self.state_model.state = AppState.DETECTING

    def _on_start_resolution_clicked(self, resolution_method: str):
        self.configuration_model.current_resolution_method = resolution_method
        self.state_model.state = AppState.RESOLVING