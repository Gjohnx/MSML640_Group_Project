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
    
    def handle_next_step(self):
        """Handle the next step in the resolution."""
        if self.cube_model is None:
            return
        
        resolution_method_name = self.configuration_model.current_resolution_method
        if not resolution_method_name:
            return
        
        resolution_method = self.configuration_model.get_resolution_method(resolution_method_name)
        if not resolution_method:
            print("Resolution method not found!")
            return
        
        # Get the next move from the resolution method
        cube_colors = self.cube_model.colors
        move = resolution_method.solve(cube_colors)
        
        if move is None:
            print("No move available!")
            return
        
        # Apply the move to the cube
        # self._print_cube(cube_colors)
        self._apply_move(move)
        self._print_cube(self.cube_model.colors)
    
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
        
        print(f"Applied move: {move}")
    
    def _rotate_face_clockwise(self, face: np.ndarray) -> np.ndarray:
        """Rotate a 3x3 face 90 degrees clockwise."""
        return np.rot90(face, k=-1)
    
    def _rotate_face_counterclockwise(self, face: np.ndarray) -> np.ndarray:
        """Rotate a 3x3 face 90 degrees counter-clockwise."""
        return np.rot90(face, k=1)
    
    def _print_cube(self, cube: np.ndarray):
        """Print all faces of the cube in a readable format."""
        # Face mapping: 0=white(U), 1=yellow(D), 2=red(F), 3=orange(B), 4=green(R), 5=blue(L)
        face_names = {
            0: "Up (White)",
            1: "Down (Yellow)",
            2: "Front (Red)",
            3: "Back (Orange)",
            4: "Right (Green)",
            5: "Left (Blue)"
        }
        
        print("\n" + "="*50)
        print("CUBE STATE")
        print("="*50)
        
        for face_idx in range(6):
            face_name = face_names[face_idx]
            face = cube[face_idx]
            print(f"\n{face_name} (Face {face_idx}):")
            for row in range(3):
                row_str = "  ".join([f"{int(face[row, col]):2d}" for col in range(3)])
                print(f"  [{row_str}]")
        
        print("="*50 + "\n")
    
    def _rotate_R(self, cube: np.ndarray, is_prime: bool) -> np.ndarray:
        """Rotate right face (green, face 4)."""
        cube = cube.copy()
        # Rotate the right face itself
        if is_prime:
            cube[4] = self._rotate_face_counterclockwise(cube[4])
        else:
            cube[4] = self._rotate_face_clockwise(cube[4])
        
        # Rotate adjacent edges: F->U->B->D->F
        # For R clockwise: F right -> U right -> B left (reversed) -> D right (reversed) -> F right
        # For R' counter-clockwise: F right -> D right (reversed) -> B left (reversed) -> U right -> F right
        temp = cube[2, :, 2].copy()  # F right column
        if is_prime:
            # R' counter-clockwise (reverse direction): F right <- D right (reversed) <- B left (reversed) <- U right <- F right
            cube[2, :, 2] = cube[1, ::-1, 2]  # F right = D right (reversed)
            cube[1, ::-1, 2] = cube[3, ::-1, 0]  # D right (reversed) = B left (reversed)
            cube[3, ::-1, 0] = cube[0, :, 2]  # B left (reversed) = U right
            cube[0, :, 2] = temp  # U right = original F right
        else:
            # R clockwise: F right <- U right <- B left (reversed) <- D right (reversed) <- F right
            cube[2, :, 2] = cube[0, :, 2]  # F right = U right
            cube[0, :, 2] = cube[3, ::-1, 0]  # U right = B left (reversed)
            cube[3, ::-1, 0] = cube[1, ::-1, 2]  # B left (reversed) = D right (reversed)
            cube[1, ::-1, 2] = temp  # D right (reversed) = original F right
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
        # For L clockwise: F left -> D left -> B right (reversed) -> U left (reversed) -> F left
        # For L' counter-clockwise: F left -> U left (reversed) -> B right (reversed) -> D left -> F left
        temp = cube[2, :, 0].copy()  # F left column
        if is_prime:
            # L' counter-clockwise: F left <- U left (reversed) <- B right (reversed) <- D left <- F left
            cube[2, :, 0] = cube[0, ::-1, 0]  # F left = U left (reversed)
            cube[0, ::-1, 0] = cube[3, ::-1, 2]  # U left (reversed) = B right (reversed)
            cube[3, ::-1, 2] = cube[1, :, 0]  # B right (reversed) = D left
            cube[1, :, 0] = temp  # D left = original F left
        else:
            # L clockwise: F left <- D left <- B right (reversed) <- U left (reversed) <- F left
            cube[2, :, 0] = cube[1, :, 0]  # F left = D left
            cube[1, :, 0] = cube[3, ::-1, 2]  # D left = B right (reversed)
            cube[3, ::-1, 2] = cube[0, ::-1, 0]  # B right (reversed) = U left (reversed)
            cube[0, ::-1, 0] = temp  # U left (reversed) = original F left
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
        # For U clockwise: F top -> L top -> B top -> R top -> F top
        # For U' counter-clockwise: F top -> R top -> B top -> L top -> F top
        temp = cube[2, 0, :].copy()  # F top row
        if is_prime:
            # U' counter-clockwise: F top <- R top <- B top <- L top <- F top
            cube[2, 0, :] = cube[4, 0, :]  # F top = R top
            cube[4, 0, :] = cube[3, 0, :]  # R top = B top
            cube[3, 0, :] = cube[5, 0, :]  # B top = L top
            cube[5, 0, :] = temp  # L top = original F top
        else:
            # U clockwise: F top <- L top <- B top <- R top <- F top
            cube[2, 0, :] = cube[5, 0, :]  # F top = L top
            cube[5, 0, :] = cube[3, 0, :]  # L top = B top
            cube[3, 0, :] = cube[4, 0, :]  # B top = R top
            cube[4, 0, :] = temp  # R top = original F top
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
        # For D clockwise: F bottom -> R bottom -> B bottom -> L bottom -> F bottom
        # For D' counter-clockwise: F bottom -> L bottom -> B bottom -> R bottom -> F bottom
        temp = cube[2, 2, :].copy()  # F bottom row
        if is_prime:
            # D' counter-clockwise: F bottom <- L bottom <- B bottom <- R bottom <- F bottom
            cube[2, 2, :] = cube[5, 2, :]  # F bottom = L bottom
            cube[5, 2, :] = cube[3, 2, :]  # L bottom = B bottom
            cube[3, 2, :] = cube[4, 2, :]  # B bottom = R bottom
            cube[4, 2, :] = temp  # R bottom = original F bottom
        else:
            # D clockwise: F bottom <- R bottom <- B bottom <- L bottom <- F bottom
            cube[2, 2, :] = cube[4, 2, :]  # F bottom = R bottom
            cube[4, 2, :] = cube[3, 2, :]  # R bottom = B bottom
            cube[3, 2, :] = cube[5, 2, :]  # B bottom = L bottom
            cube[5, 2, :] = temp  # L bottom = original F bottom
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
        # For F clockwise: U bottom -> R left -> D top (reversed) -> L right (reversed) -> U bottom
        # For F' counter-clockwise: U bottom -> L right (reversed) -> D top (reversed) -> R left -> U bottom
        temp = cube[0, 2, :].copy()  # U bottom row
        if is_prime:
            # F' counter-clockwise: U bottom <- L right (reversed) <- D top (reversed) <- R left <- U bottom
            cube[0, 2, :] = cube[5, ::-1, 2]  # U bottom = L right (reversed)
            cube[5, ::-1, 2] = cube[1, 0, ::-1]  # L right (reversed) = D top (reversed)
            cube[1, 0, ::-1] = cube[4, :, 0]  # D top (reversed) = R left
            cube[4, :, 0] = temp  # R left = original U bottom
        else:
            # F clockwise: U bottom <- R left <- D top (reversed) <- L right (reversed) <- U bottom
            cube[0, 2, :] = cube[4, :, 0]  # U bottom = R left
            cube[4, :, 0] = cube[1, 0, ::-1]  # R left = D top (reversed)
            cube[1, 0, ::-1] = cube[5, ::-1, 2]  # D top (reversed) = L right (reversed)
            cube[5, ::-1, 2] = temp  # L right (reversed) = original U bottom
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
        # For B clockwise: U top -> L left (reversed) -> D bottom (reversed) -> R right -> U top
        # For B' counter-clockwise: U top -> R right -> D bottom (reversed) -> L left (reversed) -> U top
        temp = cube[0, 0, :].copy()  # U top row
        if is_prime:
            # B' counter-clockwise: U top <- R right <- D bottom (reversed) <- L left (reversed) <- U top
            cube[0, 0, :] = cube[4, :, 2]  # U top = R right
            cube[4, :, 2] = cube[1, 2, ::-1]  # R right = D bottom (reversed)
            cube[1, 2, ::-1] = cube[5, ::-1, 0]  # D bottom (reversed) = L left (reversed)
            cube[5, ::-1, 0] = temp  # L left (reversed) = original U top
        else:
            # B clockwise: U top <- L left (reversed) <- D bottom (reversed) <- R right <- U top
            cube[0, 0, :] = cube[5, ::-1, 0]  # U top = L left (reversed)
            cube[5, ::-1, 0] = cube[1, 2, ::-1]  # L left (reversed) = D bottom (reversed)
            cube[1, 2, ::-1] = cube[4, :, 2]  # D bottom (reversed) = R right
            cube[4, :, 2] = temp  # R right = original U top
        return cube

        

