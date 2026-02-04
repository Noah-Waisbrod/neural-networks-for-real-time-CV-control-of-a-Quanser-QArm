# Real-Time QArm Control with Gesture and Object Detection

This project, developed for MREN 410 Lab 3, demonstrates the use of neural networks for real-time control of a Quanser QArm. The project has two main components:

1.  **Gesture-Based Arm Control:** Utilizes MediaPipe to recognize hand gestures ("open" and "closed") for controlling the QArm's gripper. The system also tracks the hand's centroid to guide the arm's position, enabling intuitive human-robot interaction.

2.  **Real-Time Object Detection:** Employs a YOLOv8n model to detect objects in the QArm's camera feed. The system overlays bounding boxes on the detected objects in the live video stream, showcasing the arm's environmental awareness capabilities.

## Getting Started

### Prerequisites

*   Python 3 or newer
*   Quanser QArm with RealSense camera
*   The required Python libraries can be installed via pip:
    ```bash
    pip install opencv-python mediapipe ultralytics
    ```

### Running the Code

1.  **Gesture Detection:**
    To run the gesture detection and arm control script, execute the following command:
    ```bash
    python Lab3_GestureDetection.py
    ```

2.  **Object Detection:**
    To run the object detection script, execute the following command:
    ```bash
    python Lab3_NeuralNet.py
    ```

## Acknowledgments

*   The project was developed from Quanser Research Example Code by Antonio Morales.
*   The gesture recognition functionality is powered by Google's [MediaPipe](https://mediapipe.dev/).
*   The object detection is performed using the [YOLOv8](https://ultralytics.com/yolo) model.
