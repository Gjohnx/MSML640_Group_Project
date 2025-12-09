# Neural Networks

This section contains the code, assets and scripts used to generate the synthetic data, train the neural networks and test the inference of these models.

Two neural networks are used to detect the colors of the Rubik's Cube:
- Object Detection: Used to detect the box in the image where the Rubik's Cube is located.
- Color Detection: Receiving a 100x100 picture of one of the Cube faces, returns the color of the 9 tiles of the face.

## Object Detection

For more information about the steps to train the Object Detection model, please refer to the [README.md](object-detection/training/README.md) file.

## Color Detection

For more information about the steps to train the Color Detection model, please refer to the [README.md](color-detection/training/README.md) file.

## How to Run

To train the Object Detection model, run the following command:

```bash
cd neural-networks/object-detection/
python train_yolo_vertex.py
```

**Note:** This script is written to be run on Google Vertex AI. It contains dependencies on Google Cloud Storage.

```bash
cd neural-networks/color-detection/
python train_cnn.py
```

This script can be run locally or in Google Vertex AI. To run it in Vertex AI use the `config.yaml` file in the `color-detection/training` folder.