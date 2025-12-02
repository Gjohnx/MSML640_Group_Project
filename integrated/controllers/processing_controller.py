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
        
        # Connect model signals
        self.state_model.state_changed.connect(self._on_state_changed)
        self.controls_view.next_step_clicked.connect(self.handle_next_step)
        self.controls_view.prev_step_clicked.connect(self.handle_prev_step)

        self.last_move = None
        self.last_cube_colors = None

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
    
    # Handle the previous step in the resolution
    def handle_prev_step(self):
        if self.cube_model is None:
            return
        
        resolution_method_name = self.configuration_model.current_resolution_method
        if not resolution_method_name:
            return

        resolution_method = self.configuration_model.get_resolution_method(resolution_method_name)
        if not resolution_method:
            print("Resolution method not found!")
            return

        if self.last_move is None or self.last_cube_colors is None:
            print("No previous move available!")
            return
        
        resolution_method.undo()
        self.cube_model.colors = self.last_cube_colors
        self.last_move = None
        self.last_cube_colors = None

    # Handle the next step in the resolution
    def handle_next_step(self):
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
        # self._print_cube(self.cube_model.colors)
        self.last_move = move
        self.last_cube_colors = cube_colors.copy()
    
    def _apply_move(self, move: str):
        if self.cube_model is None:
            return
        
        # Get current cube state and apply move using shared utility
        cube_colors = self.cube_model.colors.copy()
        cube_colors = CubeRotations.apply_move(cube_colors, move)
        
        # Update cube model
        self.cube_model.colors = cube_colors
        
        print(f"Applied move: {move}")
    
    def _print_cube(self, cube: np.ndarray):
        # Face mapping (hardcoded.py convention): 0=U(white), 1=R(blue), 2=F(red), 3=D(yellow), 4=L(green), 5=B(orange)
        face_names = {
            0: "Up (U/White)",
            1: "Right (R/Blue)",
            2: "Front (F/Red)",
            3: "Down (D/Yellow)",
            4: "Left (L/Green)",
            5: "Back (B/Orange)"
        }
        
        print("\n" + "="*50)
        print("CUBE STATE")
        print("="*50)
        
        for face_idx in range(6):
            face_name = face_names[face_idx]
            face = cube[face_idx]
            print(f"\n{face_name} (Face {face_idx}):")
            for row in range(3):
                # Handle both character and integer types
                row_values = []
                for col in range(3):
                    val = face[row, col]
                    if isinstance(val, (str, bytes)) or (hasattr(val, 'dtype') and val.dtype.kind == 'U'):
                        # Character value
                        char_val = str(val) if not isinstance(val, bytes) else val.decode('utf-8')
                        row_values.append(f"{char_val:>2s}")
                    else:
                        # Integer value (legacy support)
                        row_values.append(f"{int(val):2d}")
                row_str = "  ".join(row_values)
                print(f"  [{row_str}]")
        
        print("="*50 + "\n")


