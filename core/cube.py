#cube
'''
Cube Class: State representation (e.g., sticker array/map). 
Methods for all 12 basic turns (R, L, U, D, F, B and prime). 
Constructor to create a solved cube. Method to apply a move sequence. 
Validation Logic: Color count checks. Solvability checks.
'''
import copy
import random
from core.notation import Face

class Cube:
    def __init__(self):
        self.state = self._solved_state()

    def print_cube(self):
        print('\t\t'+str(self.state[5][0])+'\n\t\t'+str(self.state[5][1])+'\n\t\t'+str(self.state[5][2]))   
        print(str(self.state[3][0])+' '+str(self.state[0][0])+' '+str(self.state[2][0]))
        print(str(self.state[3][1])+' '+str(self.state[0][1])+' '+str(self.state[2][1]))   
        print(str(self.state[3][2])+' '+str(self.state[0][2])+' '+str(self.state[2][2]))
        print('\t\t'+str(self.state[1][0])+'\n\t\t'+str(self.state[1][1])+'\n\t\t'+str(self.state[1][2]))
        print('\t\t'+str(self.state[4][0])+'\n\t\t'+str(self.state[4][1])+'\n\t\t'+str(self.state[4][2]))

    def _solved_state(self):
        # Represent cube as 6 faces × 9 stickers each
        return [   [['W', 'W', 'W'],
                    ['W', 'W', 'W'],
                    ['W', 'W', 'W']], #Up/white
                
                    [['G', 'G', 'G'],
                    ['G', 'G', 'G'],
                    ['G', 'G', 'G']], #front/green
                
                    [['R', 'R', 'R'],
                    ['R', 'R', 'R'],
                    ['R', 'R', 'R']], #right/red
                
                    [['O', 'O', 'O'],
                    ['O', 'O', 'O'],
                    ['O', 'O', 'O']], #left/orange
                
                    [['Y', 'Y', 'Y'],
                    ['Y', 'Y', 'Y'],
                    ['Y', 'Y', 'Y']], #down/yellow
                
                    [['B', 'B', 'B'],
                    ['B', 'B', 'B'],
                    ['B', 'B', 'B']]]

    def rotate_x(self):
        """
        Rotate entire cube around X-axis.
        """
        old_U = [row[:] for row in self.state[Face.U.value]]
        old_F = [row[:] for row in self.state[Face.F.value]]
        old_D = [row[:] for row in self.state[Face.D.value]]
        old_B = [row[:] for row in self.state[Face.B.value]]

        # remap
        self.state[Face.U.value] = old_F
        self.state[Face.F.value] = old_D
        self.state[Face.D.value] = old_B
        self.state[Face.B.value] = old_U

        # spin side faces
        self._rotate_face_clockwise(Face.R)
        self._rotate_face_counterclockwise(Face.L)

    def rotate_x_prime(self):
        old_U = [row[:] for row in self.state[Face.U.value]]
        old_F = [row[:] for row in self.state[Face.F.value]]
        old_D = [row[:] for row in self.state[Face.D.value]]
        old_B = [row[:] for row in self.state[Face.B.value]]

        self.state[Face.U.value] = old_B
        self.state[Face.F.value] = old_U
        self.state[Face.D.value] = old_F
        self.state[Face.B.value] = old_D

        self._rotate_face_counterclockwise(Face.R)
        self._rotate_face_clockwise(Face.L)

    def rotate_y(self):
        """
        Rotate entire cube around the Y-axis.
        """
        
        old_F = [row[:] for row in self.state[Face.F.value]]
        old_R = [row[:] for row in self.state[Face.R.value]]
        old_B = [row[:] for row in self.state[Face.B.value]]
        old_L = [row[:] for row in self.state[Face.L.value]]

        self.state[Face.F.value] =  [[old_L[0][2], old_L[1][2], old_L[2][2]],
                                        [old_L[0][1], old_L[1][1], old_L[2][1]],
                                        [old_L[0][0], old_L[1][0], old_L[2][0]]]
        self.state[Face.R.value] = [[old_F[0][2], old_F[1][2], old_F[2][2]],
                                    [old_F[0][1], old_F[1][1], old_F[2][1]],
                                    [old_F[0][0], old_F[1][0], old_F[2][0]]]
        self.state[Face.B.value] = [[old_R[0][2], old_R[1][2], old_R[2][2]],
                                    [old_R[0][1], old_R[1][1], old_R[2][1]],
                                    [old_R[0][0], old_R[1][0], old_R[2][0]]]
        self.state[Face.L.value] = [[old_B[0][2], old_B[1][2], old_B[2][2]],
                                    [old_B[0][1], old_B[1][1], old_B[2][1]],
                                    [old_B[0][0], old_B[1][0], old_B[2][0]]]

        # Rotate U and D to maintain orientation
        self._rotate_face_counterclockwise(Face.U)
        self._rotate_face_clockwise(Face.D)

    def rotate_y_prime(self):
        old_F = [row[:] for row in self.state[Face.F.value]]
        old_R = [row[:] for row in self.state[Face.R.value]]
        old_B = [row[:] for row in self.state[Face.B.value]]
        old_L = [row[:] for row in self.state[Face.L.value]]

        self.state[Face.F.value] = [[old_R[2][0], old_R[1][0], old_R[0][0]],
                                    [old_R[2][1], old_R[1][1], old_R[0][1]],
                                    [old_R[2][2], old_R[1][2], old_R[0][2]]]
        self.state[Face.R.value] = [[old_B[2][0], old_B[1][0], old_B[0][0]],
                                    [old_B[2][1], old_B[1][1], old_B[0][1]],
                                    [old_B[2][2], old_B[1][2], old_B[0][2]]]
        self.state[Face.B.value] = [[old_L[2][0], old_L[1][0], old_L[0][0]],
                                    [old_L[2][1], old_L[1][1], old_L[0][1]],
                                    [old_L[2][2], old_L[1][2], old_L[0][2]]]
        self.state[Face.L.value] = [[old_F[2][0], old_F[1][0], old_F[0][0]],
                                    [old_F[2][1], old_F[1][1], old_F[0][1]],
                                    [old_F[2][2], old_F[1][2], old_F[0][2]]]

        self._rotate_face_clockwise(Face.U)
        self._rotate_face_counterclockwise(Face.D)

    # ---------- Helper rotations ----------
    def _rotate_face_clockwise(self, face):
        s = self.state[face.value]
        self.state[face.value] = [[s[2][0], s[1][0], s[0][0]],
                            [s[2][1], s[1][1], s[0][1]],
                            [s[2][2], s[1][2], s[0][2]]]

    def _rotate_face_counterclockwise(self, face):
        s = self.state[face.value]
        self.state[face.value] = [[s[0][2], s[1][2], s[2][2]],
                            [s[0][1], s[1][1], s[2][1]],
                            [s[0][0], s[1][0], s[2][0]]]

    # ---------- Rotations ----------
    def turn_R(self):
        # Rotate R face itself
        self._rotate_face_clockwise(Face.R)

        # Save edges
        u = self.state[Face.U.value]
        f = self.state[Face.F.value]
        d = self.state[Face.D.value]
        b = self.state[Face.B.value]

        # Move the edge stickers on adjacent faces
        temp = [u[0][2], u[1][2], u[2][2]]
        u[0][2], u[1][2], u[2][2] = f[0][2], f[1][2], f[2][2]
        f[0][2], f[1][2], f[2][2] = d[0][2], d[1][2], d[2][2]
        d[0][2], d[1][2], d[2][2] = b[0][2], b[1][2], b[2][2]
        b[0][2], b[1][2], b[2][2] = temp

    def turn_R_prime(self):
        # Rotate R face itself
        self._rotate_face_counterclockwise(Face.R)

        # Save edges
        u = self.state[Face.U.value]
        f = self.state[Face.F.value]
        d = self.state[Face.D.value]
        b = self.state[Face.B.value]

        # Move the edge stickers on adjacent faces
        temp = [u[0][2], u[1][2], u[2][2]]
        u[0][2], u[1][2], u[2][2] = b[0][2], b[1][2], b[2][2]
        b[0][2], b[1][2], b[2][2] = d[0][2], d[1][2], d[2][2]
        d[0][2], d[1][2], d[2][2] = f[0][2], f[1][2], f[2][2]
        f[0][2], f[1][2], f[2][2] = temp

    def turn_L(self):
        self._rotate_face_clockwise(Face.L)
        u, f, d, b = self.state[Face.U.value], self.state[Face.F.value], self.state[Face.D.value], self.state[Face.B.value]
        temp = [u[0][0], u[1][0], u[2][0]]
        u[0][0], u[1][0], u[2][0] = b[0][0], b[1][0], b[2][0]
        b[0][0], b[1][0], b[2][0] = d[0][0], d[1][0], d[2][0]
        d[0][0], d[1][0], d[2][0] = f[0][0], f[1][0], f[2][0]
        f[0][0], f[1][0], f[2][0] = temp

    def turn_L_prime(self):
        self._rotate_face_counterclockwise(Face.L)
        u, f, d, b = self.state[Face.U.value], self.state[Face.F.value], self.state[Face.D.value], self.state[Face.B.value]
        temp = [u[0][0], u[1][0], u[2][0]]
        u[0][0], u[1][0], u[2][0] = f[0][0], f[1][0], f[2][0]
        f[0][0], f[1][0], f[2][0] = d[0][0], d[1][0], d[2][0]
        d[0][0], d[1][0], d[2][0] = b[0][0], b[1][0], b[2][0]
        b[0][0], b[1][0], b[2][0] = temp

    def turn_U(self):
        self._rotate_face_clockwise(Face.U)
        f, r, b, l = self.state[Face.F.value], self.state[Face.R.value], self.state[Face.B.value], self.state[Face.L.value]
        temp = [f[0][0], f[0][1], f[0][2]]
        f[0][0], f[0][1], f[0][2] = r[2][0], r[1][0], r[0][0]
        r[2][0], r[1][0], r[0][0] = b[2][2], b[2][1], b[2][0]
        b[2][2], b[2][1], b[2][0] = l[0][2], l[1][2], l[2][2]
        l[0][2], l[1][2], l[2][2] = temp

    def turn_U_prime(self):
        self._rotate_face_counterclockwise(Face.U)
        f, r, b, l = self.state[Face.F.value], self.state[Face.R.value], self.state[Face.B.value], self.state[Face.L.value]
        temp = [f[0][0], f[0][1], f[0][2]]
        f[0][0], f[0][1], f[0][2] = l[0][2], l[1][2], l[2][2]
        l[0][2], l[1][2], l[2][2] = b[2][2], b[2][1], b[2][0]
        b[2][2], b[2][1], b[2][0] = r[2][0], r[1][0], r[0][0]
        r[2][0], r[1][0], r[0][0] = temp

    def turn_D(self):
        self._rotate_face_clockwise(Face.D)
        f, r, b, l = self.state[Face.F.value], self.state[Face.R.value], self.state[Face.B.value], self.state[Face.L.value]
        temp = [f[2][0], f[2][1], f[2][2]]
        f[2][0], f[2][1], f[2][2] = l[0][0], l[1][0], l[2][0]
        l[0][0], l[1][0], l[2][0] = b[0][2], b[0][1], b[0][0]
        b[0][2], b[0][1], b[0][0] = r[2][2], r[1][2], r[0][2]
        r[2][2], r[1][2], r[0][2] = temp

    def turn_D_prime(self):
        self._rotate_face_counterclockwise(Face.D)
        f, r, b, l = self.state[Face.F.value], self.state[Face.R.value], self.state[Face.B.value], self.state[Face.L.value]
        temp = [f[2][0], f[2][1], f[2][2]]
        f[2][0], f[2][1], f[2][2] = r[2][2], r[1][2], r[0][2]
        r[2][2], r[1][2], r[0][2] = b[0][2], b[0][1], b[0][0]
        b[0][2], b[0][1], b[0][0] = l[0][0], l[1][0], l[2][0]
        l[0][0], l[1][0], l[2][0] = temp

    def turn_F(self):
        self._rotate_face_clockwise(Face.F)
        u, r, d, l = self.state[Face.U.value], self.state[Face.R.value], self.state[Face.D.value], self.state[Face.L.value]
        temp = [u[2][0], u[2][1], u[2][2]]
        u[2][0], u[2][1], u[2][2] = l[2][0], l[2][1], l[2][2]
        l[2][0], l[2][1], l[2][2] = d[0][2], d[0][1], d[0][0]
        d[0][2], d[0][1], d[0][0] = r[2][0], r[2][1], r[2][2]
        r[2][0], r[2][1], r[2][2] = temp

    def turn_F_prime(self):
        self._rotate_face_counterclockwise(Face.F)
        u, r, d, l = self.state[Face.U.value], self.state[Face.R.value], self.state[Face.D.value], self.state[Face.L.value]
        temp = [u[2][0], u[2][1], u[2][2]]
        u[2][0], u[2][1], u[2][2] = r[2][0], r[2][1], r[2][2]
        r[2][0], r[2][1], r[2][2] = d[0][2], d[0][1], d[0][0]
        d[0][2], d[0][1], d[0][0] = l[2][0], l[2][1], l[2][2]
        l[2][0], l[2][1], l[2][2] = temp

    def turn_B(self):
        self.apply_moves("y y F y y")

    def turn_B_prime(self):
        self.apply_moves("y y F' y y")

    # ---------- Move Application ----------
    def apply_moves(self, moves: str):
        move_map = {
            "R": self.turn_R, "R'": self.turn_R_prime, "Ri": self.turn_R_prime, "R2": lambda: [self.turn_R() for _ in range(2)],
            "L": self.turn_L, "L'": self.turn_L_prime, "Li": self.turn_L_prime, "L2": lambda: [self.turn_L() for _ in range(2)],
            "U": self.turn_U, "U'": self.turn_U_prime, "Ui": self.turn_U_prime, "U2": lambda: [self.turn_U() for _ in range(2)],
            "D": self.turn_D, "D'": self.turn_D_prime, "Di": self.turn_D_prime, "D2": lambda: [self.turn_D() for _ in range(2)],
            "F": self.turn_F, "F'": self.turn_F_prime, "Fi": self.turn_F_prime, "F2": lambda: [self.turn_F() for _ in range(2)],
            "B": self.turn_B, "B'": self.turn_B_prime, "Bi": self.turn_B_prime, "B2": lambda: [self.turn_B() for _ in range(2)],
            "x": self.rotate_x, "x'": self.rotate_x_prime, "xi": self.rotate_x_prime, "x2": lambda: [self.rotate_x() for _ in range(2)],
            "y": self.rotate_y, "y'": self.rotate_y_prime, "yi": self.rotate_y_prime, "y2": lambda: [self.rotate_y() for _ in range(2)],
        }

        for mv in moves.split():
            if mv not in move_map:
                raise ValueError(f"Unknown move: {mv}")
            fn = move_map[mv]
            # For double moves we used lambda returning a list; we don't need the list itself.
            if callable(fn):
                fn()

    def is_solved(self):
        for face in self.state:
            flat = [color for row in face for color in row]
            if not all(c == flat[0] for c in flat):
                return False
        return True

    def validate(self):
        # Check color counts, solvability constraints (parity, orientation)
        pass

    def generate_random_scramble(self, length: int = 20, seed: int | None = None) -> str:
        """
        Generate a deterministic random scramble (if seed provided).
        Avoids repeating the same face consecutively.
        """
        if seed is not None:
            random.seed(seed)

        moves = ["R", "R'", "L", "L'", "U", "U'", "D", "D'", "F", "F'", "x", "x'", "y", "y'"]
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
