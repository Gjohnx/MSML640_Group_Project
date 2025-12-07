import numpy as np
import cv2
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
        
        # Internal flag for single-shot capture
        self._capture_requested = False
        
        # Connect model signals
        self.webcam_model.frame_captured.connect(self._process_frame)
        self.state_model.state_changed.connect(self._on_state_changed)
        
        # Connect View Buttons
        self.controls_view.next_step_clicked.connect(self.handle_next_step)
        self.controls_view.prev_step_clicked.connect(self.handle_prev_step)
        self.controls_view.reset_clicked.connect(self.handle_reset)
        self.controls_view.detect_clicked.connect(self._on_detect_clicked)

        # --- NEW: Connect the Resolve Button ---
        self.controls_view.resolve_clicked.connect(self._on_resolve_clicked)

        self.last_move = None
        self.last_cube_colors = None

    def _on_detect_clicked(self):
        """Called immediately when 'Detect' is clicked."""
        print(">>> BUTTON CLICKED: Requesting Capture...")
        self._capture_requested = True

            # Clear moves from previous session 
            self.last_move = None
            self.last_cube_colors = None
            
        elif state == AppState.SOLVED:
            self.last_move = None
            self.last_cube_colors = None
            print("Resolution complete")
    
    def handle_reset(self):
        if self.cube_model:
            print("Resetting Cube Model")
            self.cube_model.colors = np.full((6,3,3),'?', dtype = str)
            self.cube_model.set_rotation(15, 15, 0)
            
            # Disable the resolve button on reset
            if self.controls_view:
                self.controls_view.disable_resolve()
                self.controls_view.faces = [] 
        
        detection_method_name = self.configuration_model.current_detection_method
        detection_method = self.configuration_model.get_detection_method(detection_method_name)
        if detection_method:
            detection_method.reset()
    
    def _process_frame(self, frame: np.ndarray):

        if self.state_model.state == AppState.DETECTING:
        
            detection_method_name = self.configuration_model.current_detection_method
            detection_method = self.configuration_model.get_detection_method(detection_method_name)

            processed_frame, cube_colors, rotation = detection_method.process(frame)
            if self.cube_model is not None:
                self.cube_model.colors = cube_colors
                # Update rotation if provided by detection method
                if rotation is not None:
                    self.cube_model.set_rotation(rotation[0], rotation[1], rotation[2])
            self.view.display_frame(processed_frame)

            self.state_model.state = AppState.WAITING_FOR_DETECTION
    
    # Handle the previous step in the resolution
    def handle_prev_step(self):
        if self.cube_model is None:
            return
        
        resolution_method_name = self.configuration_model.current_resolution_method
        if not resolution_method_name:
            return

        processed_frame, full_cube_state, rotation = detection_method.process(frame)
        
        # 1. Extract Center Sticker (For Visual Feedback Text only)
        # We still find the 'primary' face just to show status text
        center_raw = '?'
        for i in range(6):
            if full_cube_state[i][1, 1] != '?':
                center_raw = str(full_cube_state[i][1, 1])
                break # Just for display text, we stop at the first one
        
        # 2. Status Text
        h, w = processed_frame.shape[:2]
        if self.state_model.state == AppState.DETECTED:
             cv2.putText(processed_frame, "Cube Complete!", (10, 30), 
                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        elif center_raw == '?':
            cv2.putText(processed_frame, "Align Grid / Center Unknown", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(processed_frame, f"Ready. Center: {center_raw}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 3. CAPTURE LOGIC (Updated for Multiple Faces/Random)
        if self._capture_requested:
            print(f">>> Processing Capture Request.")
            
            faces_found_count = 0

            # LOOP through ALL 6 potential faces in the state
            for i in range(6):
                face_data = full_cube_state[i]
                
                # If this face has valid data (center is not '?'), merge it
                if face_data[1, 1] != '?':
                    self._merge_face(i, face_data)
                    faces_found_count += 1
            
            if faces_found_count > 0:
                # SUCCESS
                # Update Rotation if method provided it
                if rotation is not None:
                    self.cube_model.set_rotation(rotation[0], rotation[1], rotation[2])
                
                cv2.putText(processed_frame, "CAPTURED!", (w//2 - 100, h//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 5)
            else:
                # FAILURE
                print("Capture ignored: No valid faces found.")
                cv2.putText(processed_frame, "CAPTURE FAILED", (10, h - 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Reset the flag so we don't capture again automatically
            self._capture_requested = False
            
            # Reset state just in case
            if self.state_model.state == AppState.DETECTING:
                self.state_model.state = AppState.WAITING_FOR_DETECTION

        self.view.display_frame(processed_frame)

    def _merge_face(self, target_index: int, new_face: np.ndarray):
        """Merges new face, updates view data, and enables resolve if full."""
        if target_index < 0 or target_index > 5:
            return

        current_total_state = self.cube_model.colors.copy()
        current_total_state[target_index] = new_face
        self.cube_model.colors = current_total_state
        
        # Calculate how many faces are actually filled (not '?')
        captured_indices = [i for i in range(6) if current_total_state[i][1,1] != '?']
        print(f"SUCCESS: Updated Face {target_index}. Faces Captured: {len(captured_indices)}/6")

        # --- UPDATED LOGIC ---
        # 1. Update the View's data so the 'Resolve' button validation works
        self.controls_view.faces = captured_indices 

        # 2. Check if complete
        if len(captured_indices) == 6:
            print(">>> CUBE COMPLETED. Enabling Resolve button.")
            self.controls_view.enable_resolve()

    def handle_prev_step(self):
        self._handle_step_action(undo=True)

    def handle_next_step(self):
        self._handle_step_action(undo=False)

    def _handle_step_action(self, undo: bool):
        if not self.cube_model: return
        
        resolution_method_name = self.configuration_model.current_resolution_method
        if not resolution_method_name: return
        
        resolution_method = self.configuration_model.get_resolution_method(resolution_method_name)
        if not resolution_method: return
        
        if undo:
            if self.last_move is None or self.last_cube_colors is None: return
            resolution_method.undo()
            self.cube_model.colors = self.last_cube_colors
            self.last_move = None
            self.last_cube_colors = None
        else:
            cube_colors = self.cube_model.colors
            move = resolution_method.solve(cube_colors)
            if move is None: return
            self._apply_move(move)
            self.last_move = move
            self.last_cube_colors = cube_colors.copy()
    
    def _apply_move(self, move: str):
        if self.cube_model is None: return
        cube_colors = self.cube_model.colors.copy()
        cube_colors = CubeRotations.apply_move(cube_colors, move)
        self.cube_model.colors = cube_colors