import numpy as np
import sys
import os
import traceback
from typing import Optional, List

# Add the project root to sys.path to allow importing from 'core'
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from core.cube import Cube
from core.beginner_solver import BeginnerSolver
from .base import ResolutionMethod

class DummyResolutionMethod(ResolutionMethod):
    """
    Adapts the Core BeginnerSolver to the Integrated ResolutionMethod interface.
    
    This class handles:
    1. Converting the UI's numpy cube state to the Core's list-based cube state.
    2. Mapping colors and face orientations (UI Red-Front vs Core Green-Front).
    3. Applying necessary face rotations required by the BeginnerSolver's logic.
    4. Running the solver.
    5. Translating solution moves back to the UI's coordinate system.
    """

    DEMO_FALLBACK = ["R", "U", "R'", "U'", "R", "U", "R'", "U'"]

    def __init__(self):
        self.solution_moves: List[str] = []
        self.current_move_index = 0
        self.solved = False

    def solve(self, cube_colors: np.ndarray) -> Optional[str]:
        # Generate solution if not already done
        if not self.solution_moves:
            if np.any(cube_colors == '?'):
                print("Cannot solve: Cube has unknown tiles.")
                return None
            
            try:
                # 1. Convert Integrated state to Core state (Remap + Rotate Faces)
                core_cube = self._convert_to_core_cube(cube_colors)
                
                # 2. Run the Beginner Solver
                solver = BeginnerSolver(core_cube)
                raw_moves = solver.solve()
                print(raw_moves)

                # 3. Translate moves from Core frame to Integrated frame
                self.solution_moves = self._translate_moves(raw_moves)
                print(f"Beginner Solution (Translated): {self.solution_moves}")
                
            except Exception as e:
                print(f"Solver failed: {e}")
                # traceback.print_exc()
                print("Falling back to DEMO sequence to prevent crash.")
                self.solution_moves = self.DEMO_FALLBACK

            self.current_move_index = 0

        # Return the next move
        if self.current_move_index < len(self.solution_moves):
            move = self.solution_moves[self.current_move_index]
            self.current_move_index += 1
            
            if self.current_move_index >= len(self.solution_moves):
                self.solved = True
                
            return move
        else:
            return None

    def undo(self):
        if self.current_move_index > 0:
            self.current_move_index -= 1
            self.solved = False

    def _convert_to_core_cube(self, ui_colors: np.ndarray) -> Cube:
        """
        Maps UI Cube (Red Front, Upright faces) to Core Cube (Green Front, Rotated faces).
        
        Face Transformations:
        - U (Up):    Rotated 90 CCW (Aligns Green to Front)
        - D (Down):  Rotated 90 CW  (Aligns Green to Front)
        - F (Green): Upright        (From UI Left)
        - R (Red):   Rotated 90 CCW (From UI Front; Down becomes Right edge)
        - B (Blue):  Rotated 180    (From UI Right; Down becomes Top edge)
        - L (Orange):Rotated 90 CW  (From UI Back;  Down becomes Left edge)
        """
        c = Cube()
        
        # Color mapping: Int chars -> Core chars
        color_map = {
            'U': 'W', 'D': 'Y', 
            'F': 'R', 'B': 'O', 
            'R': 'B', 'L': 'G'
        }

        def map_face(face_grid):
            return [[color_map.get(face_grid[row, col], '?') for col in range(3)] for row in range(3)]
        
        # core up
        c.state[0] = map_face(ui_colors[0])
        
        # core front
        c.state[1] = map_face(ui_colors[2])

        # core right
        c.state[2] = map_face(np.rot90(ui_colors[1], k=1))

        # core left
        c.state[3] = map_face(np.rot90(ui_colors[4], k=-1))

        # core down
        c.state[4] = map_face(ui_colors[3])

        # core back
        c.state[5] = map_face(np.rot90(ui_colors[5], k=2))
        c.print_cube()
        print(ui_colors)
        return c

    def _translate_moves(self, core_moves: List[str]) -> List[str]:
        """
        Translates moves from the Core notation to the Integrated system notation
        """
        translated = []
        
        for move in core_moves:
            if not move: continue
            base = move[0]
            modifier = move[1:]
            new_modifier = modifier if modifier != 'i' else "'"
            translated.append(f"{base}{new_modifier}")
            
        return translated
        # """
        # Translates moves from the Core coordinate system (Green Front) 
        # back to the Integrated system (Red Front).
        # """
        # Static map based on y' rotation (Green becomes Front)
        # Core Move -> UI Move
        # move_map = {
        #     'F': 'L', # Core Front was UI Left
        #     'R': 'F', # Core Right was UI Front
        #     'B': 'R', # Core Back was UI Right
        #     'L': 'B', # Core Left was UI Back
        #     'U': 'U', # Up is still Up
        #     'D': 'D'  # Down is still Down
        # }
        
        # # Dynamic mapping to handle 'y' rotations generated by the solver
        # current_map = move_map.copy()
        
        # for move in core_moves:
        #     if not move: continue
            
        #     base = move[0]
        #     modifier = move[1:]
            
        #     # Handle 'y' rotations (Solver re-orienting the cube)
        #     if base.lower() == 'y':
        #         # y (CW) permutes the map: F->R, R->B, B->L, L->F
        #         old_map = current_map.copy()
        #         if modifier == '' or modifier == ' ': 
        #             current_map['F'] = old_map['R']
        #             current_map['R'] = old_map['B']
        #             current_map['B'] = old_map['L']
        #             current_map['L'] = old_map['F']
        #         elif modifier == "'" or modifier == "i":
        #             current_map['F'] = old_map['L']
        #             current_map['L'] = old_map['B']
        #             current_map['B'] = old_map['R']
        #             current_map['R'] = old_map['F']
        #         elif modifier == "2":
        #             current_map['F'] = old_map['B']
        #             current_map['B'] = old_map['F']
        #             current_map['R'] = old_map['L']
        #             current_map['L'] = old_map['R']
        #         # Do not output 'y' to UI, as UI camera is static
        #         continue

        #     if base in current_map:
        #         new_base = current_map[base]
        #         new_modifier = modifier if modifier != 'i' else "'"
        #         translated.append(f"{new_base}{new_modifier}")
        #     else:
        #         translated.append(move) 
                
        # return translated