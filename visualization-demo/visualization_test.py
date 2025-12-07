from visualization import visulization_utils as vis

def main():
    vis.initialize_scene()
    vis.build_cube(size=3)
    # This is just an example command sequence. We will replace it with actual solver result moves later.
    vis.moves = ["R", "U'", "F2", "L", "D'", "R2", "U", "F'", "L2"]
    vis.add_controls()
    vis.wait_for_exit()

# Call the main and run
if __name__ == "__main__":
    main()