# Rubik's Cube Solver

## Installation and Running the Application

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Application

```bash
python main.py
```

## Architecture

The application follows a lightweight MVC pattern using PySide6's signal/slot mechanism:

- **Models**: Hold application state and emit signals when data changes
- **Views**: UI widgets that display data and emit signals for user interactions
- **Controllers**: Business logic that connects models to views using signals/slots

## Models

**WebcamModel**

Only contains the frame_captured signal to contain the information that a new frame has been captured.

**ConfigurationModel**

Contains the algorithm_changed signal to notify when a new algorithm has been selected.

**CubeModel**

Contains the state of the cube, the colors tensor, and the rotation angles. Notifies when the colors or rotation angles change by emitting the rotation_changed and cube_state_changed signals.

## Views

**MainWindow**

Contains the window and the views for the webcam, processed frame, controls, and the cube visualization. When the window is closed, it calls the cleanup callback to notify the AppController to stop the application.

**WebcamView**

Displays the webcam feed.

**ProcessedView**

Displays the processed frame.

**ControlsView**

Displays the controls for the algorithm selection.

**CubeView**

Displays the cube visualization.

## Controllers

**WebcamController**

Starts the Webcam capture thread which runs on a separate thread. The WebcamThread object is in the same file. The thread communicates with the WebcamController through signals.

Listens to **frame_ready** signal to notify the WebcamModel to update the frame. Also listens to **error_occurred** signal if there is an error with the webcam.

Provides the **start_capture** and **stop_capture** methods to be managed by the AppController, which indicates when the processing starts and ends.

**WebcamThread**

It's part of the WebcamController. It communicates with the WebcamController through signals. It implements the QT object QThread.

**ProcessingController**

Listens to the **frame_captured** signal so, when a new frame is captured, it processes it with the selected algorithm and updates the processed video in the processed view.

It accesses the WebcamModel to get the frame and the ConfigurationModel to get the selected algorithm.

**CubeController**

Listens to the **rotation_changed** and **cube_state_changed** signals from the CubeModel and updates the view accordingly.

## Signals

**frame_ready**

Emitted by the WebcamThread when a new frame is ready. Received by the WebcamController to notify the WebcamModel to update the frame. This derives on the **frame_captured** signal later.

**error_occurred**

Emitted by the WebcamThread when an error occurs. The WebcamController captures it and shows it in the view window.

**frame_captured**

Emitted by the WebcamModel when a new frame is captured. Received by the WebcamView to display the frame. Received by the ProcessingController to process the frame.

**algorithm_changed**

Emitted by the ConfigurationModel when a new algorithm has been selected. 

**rotation_changed**

Emitted by the CubeModel when the rotation angles change. Received by the CubeController to update the rotation of the cube.

**cube_state_changed**

Emitted by the CubeModel when the cube state changes. Received by the CubeController to update the colors of the cube.

**algorithm_selected**

Emitted by the ControlsView when a new algorithm has been selected. Received by the ControlsController to update the selected algorithm in the ConfigurationModel.

# References

https://rubiks.fandom.com/wiki/Notation
https://medium.com/@brad.hodkinson2/writing-code-to-solve-a-rubiks-cube-7bf9c08de01f
https://www.sfu.ca/~jtmulhol/math302/puzzles-rc.html
