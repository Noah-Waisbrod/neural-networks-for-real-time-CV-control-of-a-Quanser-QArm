"""
    MREN 410 Lab3 NeuralNetwork
    Developed from Quanser Research Example Code by Antonio Morales
    Last Updated: 08/15/2025

    References:
    https://mediapipe.readthedocs.io/en/latest/solutions/hands.html
    https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
    https://arxiv.org/abs/2006.10214
"""
# -- -- -- -- -- -- -- -- -- -- -- SETUP: READ BUT DONT CHANGE -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
# Imports
from pal.products.qarm import QArmRealSense
from pal.products.qarm import QArm
from hal.products.qarm import QArmUtilities
from hal.utilities.image_processing import ImageProcessing
import numpy as np
import cv2
import mediapipe as mp

# -- -- -- -- -- -- -- -- -- -- -- SETUP: READ BUT DONT CHANGE -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
# Parameters
imageWidth = 640
imageHeight = 480

# MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
FINGERS = {
    "thumb":  [1, 2, 3, 4],
    "index":  [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring":   [13, 14, 15, 16],
    "pinky":  [17, 18, 19, 20],
}

# -- -- -- -- -- -- -- -- -- -- -- SETUP: READ BUT DONT CHANGE -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
# Main Loop
def main():
    Ry = 0
    Rx = -0.8
    Rz = 0.8

    # Objects being set up
    myArm = QArm(hardware=1)
    myArmUtilities = QArmUtilities()
    camTool = ImageProcessing()
    np.set_printoptions(precision=2, suppress=True)

    # Initial variable states
    gripCmd = 0
    phiCmd = [0.0,-0.8,0.8,0.0]
    ledCmd = np.array([0, 1, 0], dtype=np.float64)
    cx, cy = 0, 0
    gesture = "unknown"
    last_gesture = "unknown"
    gesture_frames = 0

    with QArmRealSense(mode='RGB&DEPTH',
                       hardware=1,
                       deviceID=0,
                       frameWidthRGB=imageWidth,
                       frameHeightRGB=imageHeight,
                       frameWidthDepth=imageWidth,
                       frameHeightDepth=imageHeight,
                       readMode=1) as myCam1, \
        mp_hands.Hands(max_num_hands=1,
                        min_detection_confidence=0.45,
                        min_tracking_confidence=0.25) as hands:
        try:
          while myArm.status:                
                #---------------- SECTION 1.0 NEURAL NET ------------------------
                # Read Image Buffer and Convert to RGB image
                myCam1.read_RGB()
                img = myCam1.imageBufferRGB
                img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Feed image into MediaPipe Hands Neural Net
                result = hands.process(img_RGB)

                # Keep a clean original copy of buffer to annotate
                img_buffer_copy = img.copy()

                # Analyze result of neural network
                if result.multi_hand_landmarks:
                    # Collect landmarks for first hand detected
                    Hand_landmarks = result.multi_hand_landmarks[0]

                    # Draw landmarks cause its cool to look at 
                    mp_drawing.draw_landmarks(img_buffer_copy, Hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
                    # Convert landmarks to points
                    points = np.array([[int(lm.x*imageWidth), int(lm.y*imageHeight)] for lm in Hand_landmarks.landmark], np.float32)
                    points_norm = ((points - points[0])/ (np.linalg.norm((points - points[0]), axis=1).max() + 1e-6))

                    # Determine number of extended fingers based on landmark locations
                    extended = 0
                    for f in FINGERS:
                        f_tip = points_norm[FINGERS[f][-1]]
                        f_base = points_norm[FINGERS[f][0]]
                        if np.linalg.norm(f_tip) > np.linalg.norm(f_base) + 0.08:
                            extended += 1

                    # Determine gesture based on number of extended fingers
                    gest = None
                    if extended >= 4:
                        gest = "open"
                    else:
                        gest = "closed"

                    # Log gesture

                    # Calculate cetroid of hand
                    cv2.imwrite("3.png", img_buffer_copy)

                    cx = int(np.mean(points[:, 0]))
                    cy = int(np.mean(points[:, 1]))
                    cv2.circle(img_buffer_copy, (cx, cy), 10, (0,0,127), -1)
 
                    # Draw centroid to out annotation image

                    # Show annotations
                    cv2.imshow("Centroid", img_buffer_copy)
                    cv2.imwrite("4.png", img_buffer_copy)
                else:
                    print("No Hand Detected")

                #---------------- SECTION 2.0 ARM TRACKING -----------------------

                ey = abs(((imageWidth / 2) - cx)/(imageWidth / 2))
                ez = abs(((imageHeight // 2) - cy)/(imageHeight // 2))

                incy = 0.04*(ey)
                incz = 0.04*(ez)

                print(ez, ey)

                if cx > imageWidth // 2:
                    Ry -= incy
                else:
                    Ry += incy

                if cy > imageHeight // 2:
                    Rz += incz
                else:
                    Rz -= incz

                if Ry > 1.5:
                    Ry = 1.5
                elif Ry < -1.5:
                    Ry = -1.5

                if Rz > 1.5:
                    Rz = 1.5
                elif Rz < 0:
                    Rz = 0

                


                #---------------- SECTION 3.0 GRIPPER CONTROL -----------------------
                # Assigning gesture to gripper command
                GCmd = 0
                if gest == "closed":
                        GCmd = 1

                # Write command to arm
                phiCmd = [Ry,-0.8,Rz,0.0]
                myArm.read_write_std(phiCmd, GCmd, (255, 0, 0))

                #myArm.read_write_std(phiCMD=phiCmd, gprCMD=gripCmd, baseLED=ledCmd)
                #------------------------------------------------------------------
                cv2.waitKey(1)
        except KeyboardInterrupt:
            print("\nUser interrupt detected. Terminating QArm and OpenCV...")
        finally:
            myArm.terminate()
            cv2.destroyAllWindows()
            print("Program Ended")


if __name__ == "__main__":
    main()

