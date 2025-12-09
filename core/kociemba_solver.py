#kociemba_solver
'''
Optional
solve(cube_state): 
Implements the advanced Kociemba's Two-Phase Algorithm or integrates an external library for it. 
Returns a short, optimal move sequence.
'''
import kociemba

kociemba = None

# This method utilized kociemba library to solve the cube with Kociemba method.
def solve(cube_state_string: str) -> str:

    if kociemba is None:
        raise NotImplementedError("Kociemba library not installed. Run 'pip install kociemba'")
    
    # self.state format to the string format kociemba expects (URFDLB).
    return kociemba.solve(cube_state_string)