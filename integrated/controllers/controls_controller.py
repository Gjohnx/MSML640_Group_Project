from models.configuration_model import ConfigurationModel
from views.controls_view import ControlsView


class ControlsController:
    
    def __init__(self, configuration_model: ConfigurationModel, view: ControlsView):
        self.configuration_model = configuration_model
        self.view = view
        
        # Connect view signals to model
        self.view.algorithm_selected.connect(self._on_algorithm_selected)
    
    def _on_algorithm_selected(self, algorithm_name: str):
        self.configuration_model.current_algorithm = algorithm_name if algorithm_name else None

