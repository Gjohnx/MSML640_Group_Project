'''
read_json_state(filepath): Loads a CubeState from a JSON file. 
export_json_state(cube_state, filepath): Writes a CubeState to a JSON file.
'''
import json
from core.cube import Cube

def read_json_state(filepath: str) -> Cube:
    """Loads a CubeState from a JSON file."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        cube = Cube()
        # Assuming the JSON structure matches the internal state list
        if 'state' in data:
            cube.state = data['state']
        else:
            raise ValueError("Invalid JSON format: 'state' key missing.")
            
        return cube
    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
        return None

def export_json_state(cube: Cube, filepath: str):
    """Writes a CubeState to a JSON file."""
    data = {
        'state': cube.state
    }
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)