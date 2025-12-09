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
        self.controls_view.reset_clicked.connect(self.handle_reset)

        self.last_move = None
        self.last_cube_colors = None

        self.face_order = ['U', 'R', 'F', 'D', 'L', 'B']
        self.face_index = 0
        self.scanned_faces = [None] * 6
        self.raw_colors = [None] * 6
        self.center_colors = {}
        self.full_cube_ready = False
    
    # Extract 9 center-region color samples from 3x3 grid
    def extract_3x3_raw_colors(self, img):
        if img is None:
            return None
        H, W = img.shape[:2]
        tile = W // 3
        samples = []

        patch = 20
        half = patch // 2

        for r in range(3):
            for c in range(3):
                cx = c * tile + tile // 2
                cy = r * tile + tile // 2
                x1 = max(cx - half, 0)
                y1 = max(cy - half, 0)
                x2 = min(cx + half, W)
                y2 = min(cy + half, H)
                region = img[y1:y2, x1:x2]
                b, g, r_ = np.mean(region.reshape(-1, 3), axis=0)
                samples.append((b, g, r_))
        return samples
    
    # Store scanned face image and its raw 3x3 colors
    def store_scanned_face(self, processed_img):
        if processed_img is None:
            return False
        idx = self.face_index
        self.scanned_faces[idx] = processed_img
        samples = self.extract_3x3_raw_colors(processed_img)
        self.raw_colors[idx] = samples
        print(f"\nRaw 3×3 colors for face {self.face_order[idx]} (index {idx}):")
        for r in range(3):
            row_vals = samples[3*r : 3*r+3]
            print("   ", row_vals)
        print()
        face_label = self.face_order[idx]
        center_color = samples[4]
        self.center_colors[face_label] = center_color
        self.face_index += 1
        return self.face_index >= 6
    
    # Classify all 54 stickers into labels U,R,F,D,L,B
    def classify_all_faces(self):
        import numpy as np
        cube_labels = np.full((6, 3, 3), '?', dtype=str)
        # Center color reference
        ref = {}
        for face in self.face_order:
            if face == 'U':
                # Manually define the white color
                ref[face] = np.array([255, 255, 255], dtype=np.float32)
            else:
                ref[face] = np.array(self.center_colors[face], dtype=np.float32)
        # detect the other 54 faces
        for face_idx in range(6):
            samples = self.raw_colors[face_idx] 
            for i, (b, g, r) in enumerate(samples):
                sample_vec = np.array([b, g, r], dtype=np.float32)
                best_face = None
                best_dist = float('inf')
                for face in self.face_order:
                    d = np.linalg.norm(sample_vec - ref[face])
                    if d < best_dist:
                        best_dist = d
                        best_face = face
                row = i // 3
                col = i % 3
                cube_labels[face_idx][row][col] = best_face
        return cube_labels

    # Assemble full cube
    def assemble_and_apply_cube(self):
        cube_labels = self.classify_all_faces()
        if self.cube_model is not None:
            try:
                self.cube_model.colors = cube_labels
                print("cube_model.colors successfully updated")
            except Exception as e:
                print("Failed to update cube_model.colors:", e)
        self.full_cube_ready = True
        self.face_index = 0
        self.scanned_faces = [None] * 6
        self.raw_colors = [None] * 6
        self.center_colors = {}
        return cube_labels


    def _on_state_changed(self, state: AppState):
        if state == AppState.DETECTING:
            # Reset scanning process
            if self.cube_model:
                self.cube_model.color = np.full((6,3,3),'?', dtype = str)
            self.last_move = None
            self.last_cube_colors = None
        elif state == AppState.SOLVED:
            # Clean up after solving
            self.last_move = None
            self.last_cube_colors = None
            print("Resolution complete")
    
    def handle_reset(self):
        if self.cube_model is None:
            return
        else:
            self.cube_model.colors = np.full((6,3,3),'?', dtype = str)
        detection_method_name = self.configuration_model.current_detection_method
        print("Selected method is:", detection_method_name)
        detection_method = self.configuration_model.get_detection_method(detection_method_name)
        detection_method.reset()
    
    def _process_frame(self, frame: np.ndarray):
        if self.state_model.state == AppState.DETECTING:
            detection_method_name = self.configuration_model.current_detection_method
            print("Selected method is:", detection_method_name)
            detection_method = self.configuration_model.get_detection_method(detection_method_name)
            if detection_method_name == "Processed Comparison" and self.full_cube_ready:
                return
            processed_frame, cube_colors, rotation = detection_method.process(frame)
            if detection_method_name != "Processed Comparison":
                if self.cube_model is not None:
                    if cube_colors is not None:
                        self.cube_model.colors = cube_colors
                    if rotation is not None:
                        rx, ry, rz = rotation
                        self.cube_model.set_rotation(rx, ry, rz)
                self.view.show_single_view()
                self.view.display_frame(processed_frame)
                self.state_model.state = AppState.WAITING_FOR_DETECTION
                return

            # Process the "Processed Comparison" method
            final_img = processed_frame
            self.view.display_processed_comparison(
                final_img,
                final_img,
                final_img, 
                final_img
            )
            scan_done = self.store_scanned_face(final_img)
            print(f"[SCAN] Stored face index {self.face_index - 1}")
            if not scan_done:
                next_face = self.face_order[self.face_index]
                print(f"[SCAN] Ready for next face: {next_face}")
                self.state_model.state = AppState.WAITING_FOR_DETECTION
                return

            # ALL 6 FACES SCANNED → assemble cube!
            print("[SCAN] All 6 faces acquired. Building cube...")
            cube_labels = self.assemble_and_apply_cube()
            print("[SCAN] Final cube_labels (6x3x3):")
            print(cube_labels)
            self.state_model.state = AppState.WAITING_FOR_DETECTION
            return

    # Handle the resolution
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
        cube_colors = self.cube_model.colors
        move = resolution_method.solve(cube_colors)
        if move is None:
            print("No move available!")
            return
        
        # Apply the move to the cube
        self._apply_move(move)
        self.last_move = move
        self.last_cube_colors = cube_colors.copy()
    
    def _apply_move(self, move: str):
        if self.cube_model is None:
            return
        cube_colors = self.cube_model.colors.copy()
        cube_colors = CubeRotations.apply_move(cube_colors, move)
        self.cube_model.colors = cube_colors
        print(f"Applied move: {move}")
    
    def _print_cube(self, cube: np.ndarray):
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
                row_values = []
                for col in range(3):
                    val = face[row, col]
                    if isinstance(val, (str, bytes)) or (hasattr(val, 'dtype') and val.dtype.kind == 'U'):
                        char_val = str(val) if not isinstance(val, bytes) else val.decode('utf-8')
                        row_values.append(f"{char_val:>2s}")
                    else:
                        row_values.append(f"{int(val):2d}")
                row_str = "  ".join(row_values)
                print(f"  [{row_str}]")
        print("="*50 + "\n")


