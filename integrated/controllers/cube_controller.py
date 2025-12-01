from models.cube_model import CubeModel
from views.cube_view import CubeView

# This controller updates the cube view when the model is updated
class CubeController:
    
    def __init__(self, model: CubeModel, view: CubeView):
        self.model = model
        self.view = view
        
        # Connect model signals to view
        self.model.rotation_changed.connect(self._on_rotation_changed)
        self.model.cube_state_changed.connect(self._on_cube_state_changed)
        
        # Initialize view with current model state
        self.view.rotation = self.model.rotation
        self.view.colors = self.model.colors
    
    def _on_rotation_changed(self, x: float, y: float, z: float):
        self.view.rotation = (x, y, z)
    
    def _on_cube_state_changed(self):
        self.view.colors = self.model.colors
