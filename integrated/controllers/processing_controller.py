import numpy as np
from models.webcam_model import WebcamModel
from models.configuration_model import ConfigurationModel
from models.cube_model import CubeModel
from views.processed_view import ProcessedView
from models.state_model import StateModel
from models.state_model import AppState
from views.controls_view import ControlsView


class ProcessingController:
    
    def __init__(self, webcam_model: WebcamModel, configuration_model: ConfigurationModel, 
                 view: ProcessedView, cube_model: CubeModel = None, state_model: StateModel = None, controls_view: ControlsView = None):
        self.webcam_model = webcam_model
        self.configuration_model = configuration_model
        self.view = view
        self.cube_model = cube_model
        self.state_model = state_model
        self.controls_view = controls_view
        self._solution_moves = []
        self._current_step = 0
        
        # Connect model signals
        self.webcam_model.frame_captured.connect(self._process_frame)
        
        # Connect model signals
        self.state_model.state_changed.connect(self._on_state_changed)
        self.controls_view.next_step_clicked.connect(self.handle_next_step)

    def _on_state_changed(self, state: AppState):
        if state == AppState.DETECTING:
            # Do something here
            pass
        elif state == AppState.SOLVED:
            # Do something here
            pass
    
    def _process_frame(self, frame: np.ndarray):

        if self.state_model.state != AppState.DETECTING:
            return
        
        detection_method_name = self.configuration_model.current_detection_method
        detection_method = self.configuration_model.get_detection_method(detection_method_name)

        processed_frame, cube_colors, rotation = detection_method.process(frame)
        if self.cube_model is not None:
            self.cube_model.colors = cube_colors
            # Update rotation if provided by detection method
            if rotation is not None:
                self.cube_model.set_rotation(rotation[0], rotation[1], rotation[2])
        self.view.display_frame(processed_frame)
    
    def _initialize_solution(self):
        resolution_method_name = self.configuration_model.current_resolution_method
        resolution_method = self.configuration_model.get_resolution_method(resolution_method_name)
        cube_colors = self.cube_model.colors
        self._solution_moves = resolution_method.solve(cube_colors)
        self._current_step = 0
    
    def handle_next_step(self):
        """Handle the next step in the resolution."""
        if self.cube_model is None:
            return
        
        # If solution not initialized, initialize it
        if not self._solution_moves and self.configuration_model.current_resolution_method:
            self._initialize_solution()
        
        # Check if there are more moves
        if self._current_step >= len(self._solution_moves):
            # For random methods, get a new random move
            resolution_method_name = self.configuration_model.current_resolution_method
            if resolution_method_name == "Random":
                cube_colors = self.cube_model.colors
                resolution_method = self.configuration_model.get_resolution_method(resolution_method_name)
                if resolution_method:
                    new_moves = resolution_method.solve(cube_colors)
                    if new_moves:
                        self._solution_moves = new_moves
                        self._current_step = 0
                    else:
                        print("No moves available!")
                        return
                else:
                    print("Resolution method not found!")
                    return
            else:
                print("Solution complete!")
                return
        
        # Get the next move
        move = self._solution_moves[self._current_step]
        self._current_step += 1
        
        # Apply the move to the cube
        self._apply_move(move)
    
    def _apply_move(self, move: str):
        if self.cube_model is None:
            return
        
        # Get current cube state
        cube_colors = self.cube_model.colors.copy()
        
        # Determine if it's a prime (counter-clockwise) move
        is_prime = move.endswith("'")
        base_move = move.rstrip("'")
        
        # Apply the move based on notation
        # Face mapping: 0=white(U), 1=yellow(D), 2=red(F), 3=orange(B), 4=green(R), 5=blue(L)
        if base_move == "R":
            cube_colors = self._rotate_R(cube_colors, is_prime)
        elif base_move == "L":
            cube_colors = self._rotate_L(cube_colors, is_prime)
        elif base_move == "U":
            cube_colors = self._rotate_U(cube_colors, is_prime)
        elif base_move == "D":
            cube_colors = self._rotate_D(cube_colors, is_prime)
        elif base_move == "F":
            cube_colors = self._rotate_F(cube_colors, is_prime)
        elif base_move == "B":
            cube_colors = self._rotate_B(cube_colors, is_prime)
        
        # Update cube model
        self.cube_model.colors = cube_colors
        
        print(f"Applied move: {move} (step {self._current_step}/{len(self._solution_moves)})")
    
    def _rotate_face_clockwise(self, face: np.ndarray) -> np.ndarray:
        """Rotate a 3x3 face 90 degrees clockwise."""
        return np.rot90(face, k=-1)
    
    def _rotate_face_counterclockwise(self, face: np.ndarray) -> np.ndarray:
        """Rotate a 3x3 face 90 degrees counter-clockwise."""
        return np.rot90(face, k=1)
    
    def _rotate_R(self, cube: np.ndarray, is_prime: bool) -> np.ndarray:
        """Rotate right face (green, face 4)."""
        cube = cube.copy()
        # Rotate the right face itself
        if is_prime:
            cube[4] = self._rotate_face_counterclockwise(cube[4])
        else:
            cube[4] = self._rotate_face_clockwise(cube[4])
        
        # Rotate adjacent edges: F->U->B->D->F
        # Face 2 (F) right column -> Face 0 (U) right column -> Face 3 (B) left column (reversed) -> Face 1 (D) right column (reversed) -> Face 2 (F) right column
        temp = cube[2, :, 2].copy()
        if is_prime:
            cube[2, :, 2] = cube[1, :, 2]
            cube[1, :, 2] = cube[3, ::-1, 0]
            cube[3, ::-1, 0] = cube[0, :, 2]
            cube[0, :, 2] = temp
        else:
            cube[2, :, 2] = cube[0, :, 2]
            cube[0, :, 2] = cube[3, ::-1, 0]
            cube[3, ::-1, 0] = cube[1, ::-1, 2]
            cube[1, ::-1, 2] = temp
        return cube
    
    def _rotate_L(self, cube: np.ndarray, is_prime: bool) -> np.ndarray:
        """Rotate left face (blue, face 5)."""
        cube = cube.copy()
        # Rotate the left face itself
        if is_prime:
            cube[5] = self._rotate_face_counterclockwise(cube[5])
        else:
            cube[5] = self._rotate_face_clockwise(cube[5])
        
        # Rotate adjacent edges: F->D->B->U->F
        temp = cube[2, :, 0].copy()
        if is_prime:
            cube[2, :, 0] = cube[0, :, 0]
            cube[0, :, 0] = cube[3, ::-1, 2]
            cube[3, ::-1, 2] = cube[1, ::-1, 0]
            cube[1, ::-1, 0] = temp
        else:
            cube[2, :, 0] = cube[1, :, 0]
            cube[1, :, 0] = cube[3, ::-1, 2]
            cube[3, ::-1, 2] = cube[0, ::-1, 0]
            cube[0, ::-1, 0] = temp
        return cube
    
    def _rotate_U(self, cube: np.ndarray, is_prime: bool) -> np.ndarray:
        """Rotate up face (white, face 0)."""
        cube = cube.copy()
        # Rotate the up face itself
        if is_prime:
            cube[0] = self._rotate_face_counterclockwise(cube[0])
        else:
            cube[0] = self._rotate_face_clockwise(cube[0])
        
        # Rotate adjacent edges: F->L->B->R->F
        temp = cube[2, 0, :].copy()
        if is_prime:
            cube[2, 0, :] = cube[4, 0, :]
            cube[4, 0, :] = cube[3, 0, :]
            cube[3, 0, :] = cube[5, 0, :]
            cube[5, 0, :] = temp
        else:
            cube[2, 0, :] = cube[5, 0, :]
            cube[5, 0, :] = cube[3, 0, :]
            cube[3, 0, :] = cube[4, 0, :]
            cube[4, 0, :] = temp
        return cube
    
    def _rotate_D(self, cube: np.ndarray, is_prime: bool) -> np.ndarray:
        """Rotate down face (yellow, face 1)."""
        cube = cube.copy()
        # Rotate the down face itself
        if is_prime:
            cube[1] = self._rotate_face_counterclockwise(cube[1])
        else:
            cube[1] = self._rotate_face_clockwise(cube[1])
        
        # Rotate adjacent edges: F->R->B->L->F
        temp = cube[2, 2, :].copy()
        if is_prime:
            cube[2, 2, :] = cube[5, 2, :]
            cube[5, 2, :] = cube[3, 2, :]
            cube[3, 2, :] = cube[4, 2, :]
            cube[4, 2, :] = temp
        else:
            cube[2, 2, :] = cube[4, 2, :]
            cube[4, 2, :] = cube[3, 2, :]
            cube[3, 2, :] = cube[5, 2, :]
            cube[5, 2, :] = temp
        return cube
    
    def _rotate_F(self, cube: np.ndarray, is_prime: bool) -> np.ndarray:
        """Rotate front face (red, face 2)."""
        cube = cube.copy()
        # Rotate the front face itself
        if is_prime:
            cube[2] = self._rotate_face_counterclockwise(cube[2])
        else:
            cube[2] = self._rotate_face_clockwise(cube[2])
        
        # Rotate adjacent edges: U->R->D->L->U
        temp = cube[0, 2, :].copy()
        if is_prime:
            cube[0, 2, :] = cube[4, :, 0]
            cube[4, :, 0] = cube[1, 0, ::-1]
            cube[1, 0, ::-1] = cube[5, ::-1, 2]
            cube[5, ::-1, 2] = temp
        else:
            cube[0, 2, :] = cube[5, ::-1, 2]
            cube[5, ::-1, 2] = cube[1, 0, ::-1]
            cube[1, 0, ::-1] = cube[4, :, 0]
            cube[4, :, 0] = temp
        return cube
    
    def _rotate_B(self, cube: np.ndarray, is_prime: bool) -> np.ndarray:
        """Rotate back face (orange, face 3)."""
        cube = cube.copy()
        # Rotate the back face itself
        if is_prime:
            cube[3] = self._rotate_face_counterclockwise(cube[3])
        else:
            cube[3] = self._rotate_face_clockwise(cube[3])
        
        # Rotate adjacent edges: U->L->D->R->U
        temp = cube[0, 0, :].copy()
        if is_prime:
            cube[0, 0, :] = cube[5, ::-1, 0]
            cube[5, ::-1, 0] = cube[1, 2, ::-1]
            cube[1, 2, ::-1] = cube[4, :, 2]
            cube[4, :, 2] = temp
        else:
            cube[0, 0, :] = cube[4, :, 2]
            cube[4, :, 2] = cube[1, 2, ::-1]
            cube[1, 2, ::-1] = cube[5, ::-1, 0]
            cube[5, ::-1, 0] = temp
        return cube

