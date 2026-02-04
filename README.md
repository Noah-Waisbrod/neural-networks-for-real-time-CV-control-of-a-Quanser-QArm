# Real-Time QArm Control with Gesture and Object Detection

This project, developed for MREN 410 Lab 3, demonstrates the use of neural networks for real-time control of a Quanser QArm. The project has two main components:

1.  **Gesture-Based Arm Control:** Utilizes MediaPipe to recognize hand gestures ("open" and "closed") for controlling the QArm's gripper. The system also tracks the hand's centroid to guide the arm's position, enabling intuitive human-robot interaction.

2.  **Real-Time Object Detection:** Employs a YOLOv8n model to detect objects in the QArm's camera feed. The system overlays bounding boxes on the detected objects in the live video stream, showcasing the arm's environmental awareness capabilities.

## Acknowledgments

*   The project was developed from Quanser Research Example Code by Antonio Morales.
*   The gesture recognition functionality is powered by Google's [MediaPipe](https://mediapipe.dev/).
*   The object detection is performed using the [YOLOv8](https://ultralytics.com/yolo) model.
