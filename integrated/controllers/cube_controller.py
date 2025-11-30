from models.cube_model import CubeModel
from views.cube_view import CubeView


class CubeController:
    
    def __init__(self, model: CubeModel, view: CubeView):
        self.model = model
        self.view = view
        
        # Connect model signals to view
        self.model.rotation_changed.connect(self._on_rotation_changed)
        self.model.cube_state_changed.connect(self._on_cube_state_changed)
        
        # Initialize view with current model state
        x, y, z = self.model.rotation
        self.view.set_rotation(x, y, z)
    
    def _on_rotation_changed(self, x: float, y: float, z: float):
        self.view.set_rotation(x, y, z)
    
    def _on_cube_state_changed(self):
        # For now just update the view
        # In the future this could trigger a redraw with new colors
        self.view.gl_widget.update()

