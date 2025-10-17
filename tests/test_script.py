import pytest
import copy

# Assuming your files are in a structure like project/core/cube.py
# You might need to adjust the imports based on your project's Python path.
from core.cube import Cube
from core.beginner_solver import BeginnerSolver
from core.notation import Move, Face, parse_move_sequence, format_move_sequence

# --- Notation & Move Class Tests ---

@pytest.mark.parametrize("notation_str, expected_face, expected_prime, expected_double", [
    ("R", Face.R, False, False),
    ("U'", Face.U, True, False),
    ("Fi", Face.F, True, False), # Test 'i' for prime
    ("B2", Face.B, False, True),
    ("L", Face.L, False, False),
    ("D'", Face.D, True, False),
])
def test_move_parsing(notation_str, expected_face, expected_prime, expected_double):
    """Tests that valid move strings are parsed into correct Move objects."""
    move = Move(notation_str)
    assert move.face == expected_face
    assert move.is_prime == expected_prime
    assert move.is_double == expected_double

@pytest.mark.parametrize("invalid_notation", ["G", "R3", "U'2", "Fw", ""])
def test_move_parsing_invalid(invalid_notation):
    """Tests that invalid move strings raise a ValueError."""
    with pytest.raises(ValueError):
        Move(invalid_notation)

def test_move_formatting():
    """Tests that Move objects are formatted back into correct strings."""
    assert str(Move("R")) == "R"
    assert str(Move("U'")) == "U'"
    assert str(Move("F2")) == "F2"

def test_parse_move_sequence():
    """Tests the parsing of a full sequence string."""
    sequence = "R U R' U R U2 R'"
    expected_moves = [
        Move("R"), Move("U"), Move("R'"), Move("U"),
        Move("R"), Move("U2"), Move("R'")
    ]
    assert parse_move_sequence(sequence) == expected_moves

def test_parse_empty_sequence():
    """Tests that parsing an empty or whitespace string returns an empty list."""
    assert parse_move_sequence("") == []
    assert parse_move_sequence("   ") == []

def test_format_move_sequence():
    """Tests the formatting of a list of Move objects back to a string."""
    moves = [Move("L'"), Move("B"), Move("U2"), Move("F'")]
    expected_sequence = "L' B U2 F'"
    assert format_move_sequence(moves) == expected_sequence

def test_format_empty_sequence():
    """Tests that formatting an empty list returns an empty string."""
    assert format_move_sequence([]) == ""

# --- Cube Class Tests ---

def test_initialization_is_solved():
    """Tests that a new Cube instance is in the solved state."""
    cube = Cube()
    assert cube.is_solved(), "A newly created cube should be solved."

@pytest.mark.parametrize("move_func_name", [
    ("turn_R", "turn_R_prime"),
    ("turn_L", "turn_L_prime"),
    ("turn_U", "turn_U_prime"),
    ("turn_D", "turn_D_prime"),
    ("turn_F", "turn_F_prime"),
    ("turn_B", "turn_B_prime"),
])
def test_single_move_and_inverse(move_func_name):
    """
    Tests that applying a move and its inverse returns the cube to the solved state.
    This is a fundamental test for move correctness.
    """
    cube = Cube()
    move_forward, move_backward = move_func_name
    
    getattr(cube, move_forward)()
    getattr(cube, move_backward)()
    
    assert cube.is_solved(), f"Applying {move_forward} and {move_backward} should result in a solved cube."

@pytest.mark.parametrize("move_name", ["R", "L", "U", "D", "F", "B"])
def test_four_moves_return_to_solved(move_name):
    """Tests that applying the same move four times returns to the original state."""
    cube = Cube()
    cube.apply_moves(f"{move_name} {move_name} {move_name} {move_name}")
    assert cube.is_solved(), f"Applying a move ({move_name}) four times should cycle back to solved."

def test_u_turn_state_change_is_correct():
    """
    🔴 THIS TEST IS DESIGNED TO FAIL with the current code.
    It specifically checks if a U turn correctly moves the top row of the Front face
    to the top row of the Right face, exposing the bug in the turn logic.
    """
    cube = Cube()
    solved_state = cube._solved_state()
    
    front_face_top_row = solved_state[1][0]
    assert front_face_top_row == ['G', 'G', 'G']
    
    cube.turn_U()
    
    right_face_top_row_after_turn = cube.state[2][0]
    
    assert right_face_top_row_after_turn == front_face_top_row, \
        "The top row of the Right face should be the old top row of the Front face after a U turn."

def test_apply_moves_sexy_move_cycle():
    """Tests the move sequence parser with a common algorithm that cycles."""
    cube = Cube()
    sexy_move = "R U R' U'"
    full_sequence = " ".join([sexy_move] * 6)
    
    cube.apply_moves(full_sequence)
    assert cube.is_solved(), "Applying (R U R' U') six times should solve the cube."


# --- BeginnerSolver Class Tests ---

@pytest.fixture
def fresh_cube():
    """Provides a fresh, solved Cube instance for each solver test."""
    return Cube()

def test_solver_one_move_scramble(fresh_cube):
    """Tests if the solver can solve a cube scrambled with a single move."""
    scrambled_cube = copy.deepcopy(fresh_cube)
    scrambled_cube.apply_moves("R")
    
    assert not scrambled_cube.is_solved()
    
    solver = BeginnerSolver(scrambled_cube)
    solution_moves = solver.solve()
    solution_str = " ".join(solution_moves)
    
    scrambled_cube.apply_moves(solution_str)
    
    assert scrambled_cube.is_solved(), "Solver should solve a simple one-move scramble."

def test_solver_t_perm_scramble(fresh_cube):
    """Tests the solver against a full, standard PLL algorithm scramble."""
    scrambled_cube = copy.deepcopy(fresh_cube)
    t_perm = "R U R' U' R' F R2 U' R' U' R U R' F'"
    scrambled_cube.apply_moves(t_perm)
    
    solver = BeginnerSolver(scrambled_cube)
    solution_moves = solver.solve()
    
    scrambled_cube.apply_moves(" ".join(solution_moves))
    
    assert scrambled_cube.is_solved(), "Solver should solve a cube scrambled with a T-Permutation."

def test_simplify_moves_cancellation_and_combination():
    """Tests the basic logic of the move simplifier."""
    solver = BeginnerSolver(Cube())
    
    assert solver.simplify_moves(["R", "R'"]) == []
    assert solver.simplify_moves(["L", "U", "U'"]) == ["L"]
    
    assert solver.simplify_moves(["F", "F"]) == ["F2"]
    assert solver.simplify_moves(["B'", "B'"]) == ["B2"]
    assert solver.simplify_moves(["R", "R", "R"]) == ["R'"]

def test_simplify_moves_handles_single_char_moves_bug():
    """
    🔴 THIS TEST IS DESIGNED TO FAIL with the current code.
    It specifically targets the IndexError bug by passing a single-character move
    followed by a multi-character move.
    """
    solver = BeginnerSolver(Cube())
    try:
        result = solver.simplify_moves(["R", "L'"])
        assert result == ["R", "L'"]
    except IndexError:
        pytest.fail("simplify_moves raised an IndexError on a single-character move.")