from vpython import box, vector, rate, color, scene, button, wtext
import math
import time
import sys
import signal

# Initialize the Rubik's Cube data structures
cubelets = []
moves = []
move_index = 0
is_rotating = False

# Set the Rubik Cube face colors
# We will replace this with detected colors by detector later
FACE_COLORS = {
    "U": color.white,
    "D": color.yellow,
    "F": color.green,
    "B": color.blue,
    "L": color.orange,
    "R": color.red,
}

# Setup the 3D visualization WebUI
def initialize_scene():
    scene.title = "Rubik's Cube Interactive Visualization"
    scene.background = color.gray(0.1)
    scene.width, scene.height = 900, 700

# Create the basic 3x3x3 cubelet
def create_cubelet(pos, size=0.9):
    body = box(pos=pos, size=vector(size, size, size), color=color.white)
    s = size / 2 + 0.01
    stickers = [
        box(pos=body.pos + vector(0, s, 0),
            size=vector(size * 0.95, 0.02, size * 0.95),
            color=FACE_COLORS["U"]),
        box(pos=body.pos + vector(0, -s, 0),
            size=vector(size * 0.95, 0.02, size * 0.95),
            color=FACE_COLORS["D"]),
        box(pos=body.pos + vector(0, 0, s),
            size=vector(size * 0.95, size * 0.95, 0.02),
            color=FACE_COLORS["F"]),
        box(pos=body.pos + vector(0, 0, -s),
            size=vector(size * 0.95, size * 0.95, 0.02),
            color=FACE_COLORS["B"]),
        box(pos=body.pos + vector(-s, 0, 0),
            size=vector(0.02, size * 0.95, size * 0.95),
            color=FACE_COLORS["L"]),
        box(pos=body.pos + vector(s, 0, 0),
            size=vector(0.02, size * 0.95, size * 0.95),
            color=FACE_COLORS["R"]),
    ]
    return {"body": body, "stickers": stickers}

# Create the basic 3x3x3 Rubik's Cube
def build_cube(size=3):
    """Build the full 3x3x3 cube."""
    global cubelets
    cubelets.clear()
    offset = (size - 1) / 2
    for x in range(size):
        for y in range(size):
            for z in range(size):
                cubelets.append(create_cubelet(vector(x - offset, y - offset, z - offset)))

# Rotation
def rotate_cubelet(c, angle, axis, origin):
    c["body"].rotate(angle=angle, axis=axis, origin=origin)
    for s in c["stickers"]:
        s.rotate(angle=angle, axis=axis, origin=origin)

# Select cubelets in a given face layer
def select_layer(face, eps=0.15):
    if face == "R":
        return [c for c in cubelets if c["body"].pos.x > 1 - eps]
    if face == "L":
        return [c for c in cubelets if c["body"].pos.x < -1 + eps]
    if face == "U":
        return [c for c in cubelets if c["body"].pos.y > 1 - eps]
    if face == "D":
        return [c for c in cubelets if c["body"].pos.y < -1 + eps]
    if face == "F":
        return [c for c in cubelets if c["body"].pos.z > 1 - eps]
    if face == "B":
        return [c for c in cubelets if c["body"].pos.z < -1 + eps]
    return []

# Round positions to prevent the situation that it rotates "half circle"
def snap_positions():
    for c in cubelets:
        p = c["body"].pos
        c["body"].pos = vector(round(p.x, 1), round(p.y, 1), round(p.z, 1))
        for s in c["stickers"]:
            sp = s.pos
            s.pos = vector(round(sp.x, 1), round(sp.y, 1), round(sp.z, 1))

# Face Rotation
def rotate_layer(face, angle):
    global is_rotating
    layer = select_layer(face)
    if not layer:
        return

    axis_map = {
        "R": vector(1, 0, 0),
        "L": vector(-1, 0, 0),
        "U": vector(0, 1, 0),
        "D": vector(0, -1, 0),
        "F": vector(0, 0, 1),
        "B": vector(0, 0, -1),
    }
    axis = axis_map[face]
    steps = 15
    step_angle = angle / steps
    is_rotating = True
    for _ in range(steps):
        rate(60)
        for c in layer:
            rotate_cubelet(c, step_angle, axis, vector(0, 0, 0))
    snap_positions()
    is_rotating = False

# Perform a action move command in the action list
def execute_move(mv):
    """Perform a single Rubik move."""
    face = mv[0]
    angle = math.pi / 2
    if "'" in mv or "i" in mv:
        angle *= -1
    if "2" in mv:
        angle *= 2
    rotate_layer(face, angle)

# Store the next move command, which we will use it as a part of command trace
def next_move(_):
    """Button callback — step forward one move."""
    global move_index
    if is_rotating or move_index >= len(moves):
        return
    execute_move(moves[move_index])
    move_index += 1

# Store the previous move command, which we will use it as a part of command trace
def prev_move(_):
    global move_index
    if is_rotating or move_index <= 0:
        return
    move_index -= 1
    mv = moves[move_index]
    if "'" in mv or "i" in mv:
        mv = mv[0]
    else:
        mv = mv + "'"
    execute_move(mv)

# UI controls to show the previous and next move
def add_controls():
    """Add Previous / Next buttons."""
    wtext(text="\n\n")
    button(text="◀ Previous Step", bind=prev_move)
    button(text="▶ Next Step", bind=next_move)
    wtext(text="\n\n")

# This function is used to ensure that the terminal window remains open while the user are interacting with the 3D visualization.
# Also ensure that the terminal implementation will close after user exit the website.
def wait_for_exit(timeout=None):
    """Keep window open until click or Ctrl+C."""
    print("Rubik visualization running...")
    print("• Click inside the 3D window to close.")
    print("• Or press Ctrl+C in terminal to stop immediately.")
    def handle_sigint(signum, frame):
        print("\nInterrupted by user (Ctrl+C). Shutting down cleanly...")
        sys.exit(0)
    signal.signal(signal.SIGINT, handle_sigint)
    try:
        start = time.time()
        while True:
            rate(30)
            if not scene.visible:
                print("Browser window closed — exiting.")
                break
            if timeout and time.time() - start > timeout:
                print("Timeout reached — exiting.")
                break
    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected.")
    finally:
        sys.exit(0)