from models.configuration_model import ConfigurationModel
from views.controls_view import ControlsView


class ControlsController:
    
    def __init__(self, configuration_model: ConfigurationModel, view: ControlsView):
        self.configuration_model = configuration_model
        self.view = view
        
        # Connect view signals to model
        self.view.detection_method_selected.connect(self._on_detection_method_selected)
    
    def _on_detection_method_selected(self, detection_method_name: str):
        self.configuration_model.current_detection_method = detection_method_name if detection_method_name else None

