#pipeline
'''
main(): Parses CLI arguments (--scramble, --mode). 
run_pipeline(scramble, mode): Orchestrates the process (importer → solver → output). 
Handles logging and fallbacks (e.g., if Kociemba fails).
'''

import argparse
from core.beginner_solver import BeginnerSolver
from services.simulator_importer import scan_cube

def main():
    parser = argparse.ArgumentParser(description="Rubik's Cube Simulator CLI")
    parser.add_argument("--scramble", type=str, help="Scramble sequence (e.g. \"R U R' U'\")")
    #parser.add_argument("--mode", type=str, default="beginner", help="Solver mode (e.g. beginner, cfop)")
    args = parser.parse_args()

    scramble, cube = scan_cube(args.scramble)
    print(f"Scramble: {scramble}\n")

    print("Scrambled cube:\n")
    cube.print_cube()

    solver = BeginnerSolver(cube)
    solution = solver.solve()
    print(f"Solution:\n{solver.solve()}")

if __name__ == "__main__":
    main()