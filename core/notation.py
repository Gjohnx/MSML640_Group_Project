#notation
'''
parse_move_sequence(sequence: str): Converts a string like "R U R'" into a list of move objects/enums. 
format_move_sequence(moves: list): Converts a list of moves back into a standard string format.
'''
from enum import Enum
from typing import List

# U stands for Up.
# F stands for Front.
# R stands for Right.
# L stands for Left.
# D stands for Down.
# B stands for Back. 

class Face(Enum):
    U = 0
    F = 1
    R = 2
    L = 3
    D = 4
    B = 5

# Algorithem from https://jperm.net/3x3/moves
class Move:
    _face_map = {
        'U': Face.U, 'F': Face.F, 'R': Face.R,
        'L': Face.L, 'D': Face.D, 'B': Face.B
    }

    def __init__(self, notation: str):
        if not isinstance(notation, str) or not (1 <= len(notation) <= 2):
            raise ValueError(f"Invalid move notation format: '{notation}'")

        # Parse the face (e.g., 'R' from "R'")
        face_char = notation[0].upper()
        if face_char not in self._face_map:
            raise ValueError(f"Invalid face specified in move: '{face_char}'")
        self.face: Face = self._face_map[face_char]

        # Set defaults
        self.is_prime: bool = False
        self.is_double: bool = False

        if len(notation) > 1:
            modifier = notation[1]
            if modifier in ("'", "i"):
                self.is_prime = True
            elif modifier == "2":
                self.is_double = True
            else:
                raise ValueError(f"Invalid modifier in move: '{modifier}'")

    def __str__(self) -> str:
        """Formats the move object back into standard string notation."""
        modifier = ""
        if self.is_double:
            modifier = "2"
        elif self.is_prime:
            modifier = "'"
        return f"{self.face.name}{modifier}"

    def __repr__(self) -> str:
        """Provides a developer-friendly representation of the Move object."""
        return f"Move('{self.__str__()}')"

    def __eq__(self, other) -> bool:
        """Checks for equality between two Move objects, useful for testing."""
        if not isinstance(other, Move):
            return NotImplemented
        return (self.face == other.face and
                self.is_prime == other.is_prime and
                self.is_double == other.is_double)


def parse_move_sequence(sequence: str) -> List[Move]:
    if not sequence.strip():
        return []  # Return an empty list for an empty or whitespace-only string
    
    move_strings = sequence.strip().split()
    return [Move(s) for s in move_strings]


def format_move_sequence(moves: List[Move]) -> str:
    return " ".join(str(move) for move in moves)