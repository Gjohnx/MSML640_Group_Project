import numpy as np
from typing import Optional
from .base import ResolutionMethod
import kociemba


class KociembaResolutionMethod(ResolutionMethod):
    
    def __init__(self):
        self.solution_moves = []  # List of moves in the solution
        self.current_move_index = -1  # Index of the next move to execute
        self.solved = False
    
    def solve(self, cube_colors: np.ndarray) -> Optional[str]:
        # If we haven't generated the solution yet, generate it
        if not self.solution_moves:
            solution_string = self._get_kociemba_solution(cube_colors)
            if solution_string:
                # Parse the solution string into individual moves
                self.solution_moves = solution_string.split()
                self.current_move_index = 0
            else:
                return None
        
        # Return the next move in the sequence
        if self.current_move_index < len(self.solution_moves):
            move = self.solution_moves[self.current_move_index]
            self.current_move_index += 1
            
            # Check if we've reached the end of the solution
            if self.current_move_index >= len(self.solution_moves):
                self.solved = True
            
            return move
        else:
            # Already solved
            return None
    
    # Undo the last move by going back in the solution sequence
    def undo(self):
        if self.current_move_index > 0:
            self.current_move_index -= 1
            self.solved = False
    
    # Kociemba expects a string of 54 characters representing the cube state in the order: U R F D L B (Up, Right, Front, Down, Left, Back).
    def _get_kociemba_solution(self, cube_colors: np.ndarray) -> Optional[str]:
        # Check if cube has any unknown tiles
        if np.any(cube_colors == '?'):
            print("Error: Cube has unknown tiles. Complete detection first.")
            return None
        
        cube_string = self._cube_to_kociemba_string(cube_colors)
        print(f"Kociemba cube string: {cube_string}")
        
        try:
            # Get the solution from Kociemba
            solution = kociemba.solve(cube_string)
            print(f"Kociemba solution: {solution}")
            return solution
        except Exception as e:
            print(f"Error solving cube with Kociemba: {e}")
            return None
    
    # Convert cube_colors to Kociemba format string
    # Face mapping: 0=U, 1=R, 2=F, 3=D, 4=L, 5=B
    # Kociemba expects the string in URFDLB order, with each face read left-to-right, top-to-bottom
    def _cube_to_kociemba_string(self, cube_colors: np.ndarray) -> str:
        cube_string = ""
        
        # Kociemba expects faces in order: U(0), R(1), F(2), D(3), L(4), B(5)
        for face_idx in range(6):
            face = cube_colors[face_idx]
            for row in range(3):
                for col in range(3):
                    cube_string += face[row, col]
        
        return cube_string

