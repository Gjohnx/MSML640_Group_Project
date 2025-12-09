# Rubik's Cube Tutor & Solver (MSML640 Group Project)

## Members
- Jianjiong Xiang
- Damian Calabresi
- Alexios Papazoglou
- Ruikang Yan
- Xiaolong Zhu

## Prerequisites
- Python 3.13 (or 3.10+)
- Webcam
- C++ 14.0 or higher
- Python dependencies listed in `integrated/requirements.txt`

## Modules

### Integrated

This is the main application that combines the UI with the MVC pattern. To run the Rubik's Cube Solver, run this module.

For more information about the architecture of the application, please refer to the [README.md](integrated/README.md) file.

### Neural Networks

This section contains the code, assets and scripts used to generate the synthetic data, train the neural networks and test the inference of these models.

Two neural networks are used to detect the colors of the Rubik's Cube:
- Object Detection: Used to detect the box in the image where the Rubik's Cube is located.
- Color Detection: Receiving a 100x100 picture of one of the Cube faces, returns the color of the 9 tiles of the face.

For more information about the neural networks, please refer to the [README.md](neural-networks/README.md) file.

### Core

This section contains the main algorithms implemented for the resolution of the Rubik's Cube. The logic implemented by this algorithms has already been integrated into the application localed in the `integrated` folder.

This module is maintained for historical purposes.

## How to Run

### Integrated UI with PySide6

```bash
cd integrated/
pip install -r requirements.txt
python main.py
```

**Note:** For same result as *Demo*, change branch to *Baseline-1* and run the application from the `integrated` folder. The *main* branch contains the CNN based and QBR detection methods.

## Step by Step Guide

**Detection Process**
- Select the Detection Method (YOLO+CNN, HSV (Grid Rectify), QBR (Simple) are the main methods)
- Click on the "Start Detection" button to start the detection process.
- Show one face of the Rubik's Cube on the webcam.
- Click on "Detect" to capture a frame of the webcam and process it with the selected detection method.
- The detected face will be displayed in the 3D cube visualization. The processed frame may contain information useful to understand the algorithm.
- Click on "Reset" to clean the discovered colors from the cube.
- Continue clicking on "Detect" if you want to override the detected colors.

**Resolution Process**
- Resolution method option will be enabled after the cube is completely detected.
- Select the Resolution Method (Kociemba, Beginner Solver are the main methods)
- Click on the "Start Resolution" button to start the resolution process.
- Click on "Next Step" to go to the next step of the resolution. The 3D cube will move to the next state of the cube.
- Click on "Prev Step" to go to the previous step of the resolution. Only one step back is allowed.
- The expected moves using the Singmaster notation will be displayed in the console.
- Continue clicking on "Next Step" to solve the cube.
- The cube will be solved and the solution will be displayed in the 3D cube visualization.

### How to show the cube to the camera

The center tile of a cube face indicates which face of the solved cube it belongs to. The center tile is the point of reference.

A Rubik's Cube has 6 faces, each face is usually not identified by its own color, but by the position from the point of view of the camera. These positions are:
- Front
- Right
- Back
- Left
- Up
- Down

Conventions vary, but the conventions used in this project are:
- Front: Red
- Right: Blue
- Back: Orange
- Left: Green
- Up: White
- Down: Yellow

Therefore, to show the cube to the camera, you should place the cube in a way that the front face is red, the right face is blue, the back face is orange, the left face is green, the up face is white, and the down face is yellow.

The camera should be placed in a way that the cube is in the center of the field of view.

When rotation the cube, for example to show the Blue face, do it always keeping the White face up and Yellow face down. The algorithm can detect all the colors in the Face that's shown to the camera, but it can't detect the rotation of this face. The rotation of the face depends on where is the White face and the Yellow face in relation to the camera, but, as this cannot be seen by the camera, it needs to be kept constant.
