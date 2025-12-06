import numpy as np
from models.webcam_model import WebcamModel
from models.configuration_model import ConfigurationModel
from models.cube_model import CubeModel
from views.processed_view import ProcessedView
from models.state_model import StateModel
from models.state_model import AppState
from views.controls_view import ControlsView
from services.cube_rotations import CubeRotations


class ProcessingController:
    
    def __init__(self, webcam_model: WebcamModel, configuration_model: ConfigurationModel, 
                 view: ProcessedView, cube_model: CubeModel = None, state_model: StateModel = None, controls_view: ControlsView = None):
        self.webcam_model = webcam_model
        self.configuration_model = configuration_model
        self.view = view
        self.cube_model = cube_model
        self.state_model = state_model
        self.controls_view = controls_view
        
        # Connect model signals
        self.webcam_model.frame_captured.connect(self._process_frame)
        self.state_model.state_changed.connect(self._on_state_changed)
        self.controls_view.next_step_clicked.connect(self.handle_next_step)
        self.controls_view.prev_step_clicked.connect(self.handle_prev_step)

        self.last_move = None
        self.last_cube_colors = None

    def _on_state_changed(self, state: AppState):
        if state == AppState.DETECTING:
            # Reset the Cube Model for a new scan session
            if self.cube_model:
                self.cube_model.colors = np.full((6, 3, 3), '?', dtype=str)
            
            # Clear Undo History
            self.last_move = None
            self.last_cube_colors = None
            
        elif state == AppState.SOLVED:
            self.last_move = None
            self.last_cube_colors = None
            print("Resolution complete: Cube is solved.")
    
    def _process_frame(self, frame: np.ndarray):

        if self.state_model.state != AppState.DETECTING:
            return
        
        detection_method_name = self.configuration_model.current_detection_method
        detection_method = self.configuration_model.get_detection_method(detection_method_name)

        processed_frame, detected_colors, rotation = detection_method.process(frame)
        
        # --- MERGE LOGIC START ---
        if self.cube_model is not None:
            # Get current state from model (to preserve previous scans)
            current_state = self.cube_model.colors.copy()
            
            # Identify which faces in the new detection are valid (not '?')
            # The detector returns a 6x3x3 array where only the identified face is filled
            mask = detected_colors != '?'
            
            if np.any(mask):
                # Update only the valid new detections
                current_state[mask] = detected_colors[mask]
                self.cube_model.colors = current_state
            
            # Update rotation if provided by detection method
            if rotation is not None:
                self.cube_model.set_rotation(rotation[0], rotation[1], rotation[2])
        # --- MERGE LOGIC END ---
        
        self.view.display_frame(processed_frame)
    
    def handle_prev_step(self):
        if self.cube_model is None: return
        
        resolution_method_name = self.configuration_model.current_resolution_method
        if not resolution_method_name: return

        resolution_method = self.configuration_model.get_resolution_method(resolution_method_name)
        if not resolution_method: return

        if self.last_move is None or self.last_cube_colors is None: return
        
        resolution_method.undo()
        self.cube_model.colors = self.last_cube_colors
        self.last_move = None
        self.last_cube_colors = None

    def handle_next_step(self):
        if self.cube_model is None: return
        
        resolution_method_name = self.configuration_model.current_resolution_method
        if not resolution_method_name: return
        
        resolution_method = self.configuration_model.get_resolution_method(resolution_method_name)
        if not resolution_method: return
        
        cube_colors = self.cube_model.colors
        move = resolution_method.solve(cube_colors)
        
        if move is None:
            print("No move available!")
            return
        
        self._apply_move(move)
        self.last_move = move
        self.last_cube_colors = cube_colors.copy()
    
    def _apply_move(self, move: str):
        if self.cube_model is None: return
        
        cube_colors = self.cube_model.colors.copy()
        cube_colors = CubeRotations.apply_move(cube_colors, move)
        self.cube_model.colors = cube_colors
        print(f"Applied move: {move}")