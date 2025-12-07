import numpy as np


class CubeRotations:
    
    @staticmethod
    def rotate_face_clockwise(face: np.ndarray) -> np.ndarray:
        return np.rot90(face, k=-1)
    
    @staticmethod
    def rotate_face_counterclockwise(face: np.ndarray) -> np.ndarray:
        return np.rot90(face, k=1)
    
    @staticmethod
    def rotate_R(cube: np.ndarray, is_prime: bool) -> np.ndarray:
        cube = cube.copy()
        if is_prime:
            cube[1] = CubeRotations.rotate_face_counterclockwise(cube[1])
        else:
            cube[1] = CubeRotations.rotate_face_clockwise(cube[1])
        
        temp_F = cube[2, :, 2].copy()
        temp_U = cube[0, :, 2].copy()
        temp_B = cube[5, :, 0].copy()
        temp_D = cube[3, :, 2].copy()
        
        if is_prime:
            cube[3, :, 2] = temp_F
            cube[5, :, 0] = temp_D[::-1]
            cube[0, :, 2] = temp_B[::-1]
            cube[2, :, 2] = temp_U
        else:
            cube[0, :, 2] = temp_F
            cube[5, :, 0] = temp_U[::-1]
            cube[3, :, 2] = temp_B[::-1]
            cube[2, :, 2] = temp_D
        return cube
    
    @staticmethod
    def rotate_L(cube: np.ndarray, is_prime: bool) -> np.ndarray:
        cube = cube.copy()
        if is_prime:
            cube[4] = CubeRotations.rotate_face_counterclockwise(cube[4])
        else:
            cube[4] = CubeRotations.rotate_face_clockwise(cube[4])
        
        temp_F = cube[2, :, 0].copy()
        temp_U = cube[0, :, 0].copy()
        temp_B = cube[5, :, 2].copy()
        temp_D = cube[3, :, 0].copy()
        
        if is_prime:
            cube[0, :, 0] = temp_F
            cube[5, :, 2] = temp_U[::-1]
            cube[3, :, 0] = temp_B[::-1]
            cube[2, :, 0] = temp_D
        else:
            cube[3, :, 0] = temp_F
            cube[5, :, 2] = temp_D[::-1]
            cube[0, :, 0] = temp_B[::-1]
            cube[2, :, 0] = temp_U
        return cube
    
    @staticmethod
    def rotate_U(cube: np.ndarray, is_prime: bool) -> np.ndarray:
        cube = cube.copy()
        if is_prime:
            cube[0] = CubeRotations.rotate_face_counterclockwise(cube[0])
        else:
            cube[0] = CubeRotations.rotate_face_clockwise(cube[0])
        
        temp_F = cube[2, 0, :].copy()
        temp_L = cube[4, 0, :].copy()
        temp_B = cube[5, 0, :].copy()
        temp_R = cube[1, 0, :].copy()
        
        if is_prime:
            cube[1, 0, :] = temp_F
            cube[5, 0, :] = temp_R
            cube[4, 0, :] = temp_B
            cube[2, 0, :] = temp_L
        else:
            cube[4, 0, :] = temp_F
            cube[5, 0, :] = temp_L
            cube[1, 0, :] = temp_B
            cube[2, 0, :] = temp_R
        return cube
    
    @staticmethod
    def rotate_D(cube: np.ndarray, is_prime: bool) -> np.ndarray:
        cube = cube.copy()
        if is_prime:
            cube[3] = CubeRotations.rotate_face_counterclockwise(cube[3])
        else:
            cube[3] = CubeRotations.rotate_face_clockwise(cube[3])
        
        temp_F = cube[2, 2, :].copy()
        temp_R = cube[1, 2, :].copy()
        temp_B = cube[5, 2, :].copy()
        temp_L = cube[4, 2, :].copy()
        
        if is_prime:
            cube[4, 2, :] = temp_F
            cube[5, 2, :] = temp_L
            cube[1, 2, :] = temp_B
            cube[2, 2, :] = temp_R
        else:
            cube[1, 2, :] = temp_F
            cube[5, 2, :] = temp_R
            cube[4, 2, :] = temp_B
            cube[2, 2, :] = temp_L
        return cube
    
    @staticmethod
    def rotate_F(cube: np.ndarray, is_prime: bool) -> np.ndarray:
        cube = cube.copy()
        if is_prime:
            cube[2] = CubeRotations.rotate_face_counterclockwise(cube[2])
        else:
            cube[2] = CubeRotations.rotate_face_clockwise(cube[2])
        
        temp_U = cube[0, 2, :].copy()
        temp_R = cube[1, :, 0].copy()
        temp_D = cube[3, 0, :].copy()
        # Store a copy of the rightmost column of the left face for later use in the rotation
        temp_L = cube[4, :, 2].copy()
        
        if is_prime:
            cube[4, :, 2] = temp_U[::-1]
            cube[3, 0, :] = temp_L
            cube[1, :, 0] = temp_D[::-1]
            cube[0, 2, :] = temp_R
        else:
            cube[1, :, 0] = temp_U
            cube[3, 0, :] = temp_R[::-1]
            cube[4, :, 2] = temp_D
            cube[0, 2, :] = temp_L[::-1]
        return cube
    
    @staticmethod
    def rotate_B(cube: np.ndarray, is_prime: bool) -> np.ndarray:
        cube = cube.copy()
        if is_prime:
            cube[5] = CubeRotations.rotate_face_counterclockwise(cube[5])
        else:
            cube[5] = CubeRotations.rotate_face_clockwise(cube[5])
        
        temp_U = cube[0, 0, :].copy()
        temp_L = cube[4, :, 0].copy()
        temp_D = cube[3, 2, :].copy()
        temp_R = cube[1, :, 2].copy()
        
        if is_prime:
            cube[1, :, 2] = temp_U
            cube[3, 2, :] = temp_R[::-1]
            cube[4, :, 0] = temp_D
            cube[0, 0, :] = temp_L[::-1]
        else:
            cube[4, :, 0] = temp_U[::-1]
            cube[3, 2, :] = temp_L
            cube[1, :, 2] = temp_D[::-1]
            cube[0, 0, :] = temp_R
        return cube
    
    @staticmethod
    def apply_move(cube: np.ndarray, move: str) -> np.ndarray:
        # Determine move type
        is_double = move.endswith("2")
        is_prime = move.endswith("'")
        
        # Extract base move
        if is_double:
            base_move = move.rstrip("2")
        elif is_prime:
            base_move = move.rstrip("'")
        else:
            base_move = move
        
        # Apply the move
        if is_double:
            # Apply twice for double moves
            cube = CubeRotations._apply_single_move(cube, base_move, False)
            cube = CubeRotations._apply_single_move(cube, base_move, False)
        else:
            cube = CubeRotations._apply_single_move(cube, base_move, is_prime)
        
        return cube
    
    @staticmethod
    def _apply_single_move(cube: np.ndarray, base_move: str, is_prime: bool) -> np.ndarray:
        if base_move == "R":
            return CubeRotations.rotate_R(cube, is_prime)
        elif base_move == "L":
            return CubeRotations.rotate_L(cube, is_prime)
        elif base_move == "U":
            return CubeRotations.rotate_U(cube, is_prime)
        elif base_move == "D":
            return CubeRotations.rotate_D(cube, is_prime)
        elif base_move == "F":
            return CubeRotations.rotate_F(cube, is_prime)
        elif base_move == "B":
            return CubeRotations.rotate_B(cube, is_prime)
        return cube

