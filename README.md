# Rubik's Cube Tutor & Solver (MSML640 Group Project)
# Member: Jianjiong Xiang, Damian Calabresi, Alexios Papazoglou, Ruikang Yan, Xiaolong Zhu

### Prerequisites
- Python 3.13 (or 3.10+)
- Webcam
- C++ 14.0 or higher

# Modules

## Core

This section contains the main algorithms implemented for the resolution of the Rubik's Cube. The logic implemented by this algorithms has already been integrated into the application localed in the `integrated` folder.

## Integrated

This is the main application that combines the UI with the MVC pattern. To run the Rubik's Cube Solver, run this module.

For more information about the architecture of the application, please refer to the [README.md](integrated/README.md) file.

## Neural Networks

This section contains the code, assets and scripts used to generate the synthetic data, train the neural networks and test the inference of these models.

Two neural networks are used to detect the colors of the Rubik's Cube:
- Object Detection: Used to detect the box in the image where the Rubik's Cube is located.
- Color Detection: Receiving a 100x100 picture of one of the Cube faces, returns the color of the 9 tiles of the face.

# How to Run

## Integrated UI with PySide6

```bash
cd integrated
python main.py
```

**Note:** For same result as *Demo*, change branch to *Baseline-1* and run the application from the `integrated` folder. The *main* branch contains the CNN based and QBR detection methods.