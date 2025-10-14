#notation
'''
parse_move_sequence(sequence: str): Converts a string like "R U R'" into a list of move objects/enums. 
format_move_sequence(moves: list): Converts a list of moves back into a standard string format.
'''
from enum import Enum
class Face(Enum):
    U = 0
    F = 1
    R = 2
    L = 3
    D = 4
    B = 5