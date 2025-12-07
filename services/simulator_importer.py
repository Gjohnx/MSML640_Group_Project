#simulator_importer
'''
scan_cube(scramble: str = None): Main entry point. 
parse_scramble(scramble_str): Converts a move sequence string to a resulting CubeState object. 
generate_random_scramble(): Generates a valid random scramble string.
'''
# simulator/simulator_importer.py
from core.cube import Cube
import random

def generate_random_scramble(self, length: int = 20, seed: int | None = None) -> str:
    """
    Generate a deterministic random scramble (if seed provided).
    Avoids repeating the same face consecutively.
    """
    if seed is not None:
        random.seed(seed)

    moves = ["R", "R'", "L", "L'", "U", "U'", "D", "D'", "F", "F'", "B", "B'"]
    scramble = []
    last_face = None

    for _ in range(length):
        # Fallback if somehow empty (shouldn't happen, but safe)
        
        available = [m for m in moves if not m.startswith(last_face or '')]
        if not available:
            available = moves[:]
        
        move = random.choice(available)
        scramble.append(move)
        last_face = move[0]

    return " ".join(scramble)

def parse_scramble(scramble_str: str) -> Cube:
    """Take a string of moves and return the resulting Cube state."""
    cube = Cube()
    cube.apply_moves(scramble_str)
    return cube


def scan_cube(scramble: str = None):
    """
    Main entry point: create a cube, scramble it (randomly or given string),
    and return both the scramble and resulting state.
    """
    if scramble is None:
        scramble = generate_random_scramble(30)
    cube = parse_scramble(scramble)
    return scramble, cube
