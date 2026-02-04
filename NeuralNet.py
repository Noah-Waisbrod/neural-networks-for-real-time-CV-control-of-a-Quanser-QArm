"""
    MREN 410 QArm Lab3 Neural Net
    Developed from Quanser Research Example Code by Antonio Morales
    Last Updated: 05/20/2025

"""

# Imports
from pal.products.qarm import QArmRealSense
from hal.utilities.image_processing import ImageProcessing
import numpy as np
import time
import cv2
from ultralytics import YOLO


# Time
startTime = time.time()
def elapsed_time():
    return time.time() - startTime

# Image Params
imageWidth = 640
imageHeight = 480

## Initialize the RealSense camera for RGB and Depth data
with QArmRealSense(mode='RGB&DEPTH',
                   hardware=1,
                   deviceID= 0,
                   frameWidthRGB=imageWidth,
                   frameHeightRGB=imageHeight,
                   frameWidthDepth=imageWidth,
                   frameHeightDepth=imageHeight,
                   readMode = 1) as myCam1:

    t0 = time.time()

    #Image Processing
    camTool = ImageProcessing()

    # Main Loop
    try:
        while True:
            #--------------------- SECTION 1.0 ---------------------------------
            # Load model from ultralytics
            model = YOLO("yolov8n.pt")

            # Read Camera
            myCam1.read_RGB()
            img = myCam1.imageBufferRGB

            # Convert to RGB image format
            img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # eval
            result = model(img_RGB, verbose=False)

            # boxes on image
            img_box = result[0].plot()

            # Convert back to show
            img_show = img_RGB = cv2.cvtColor(img_box, cv2.COLOR_RGB2BGR)

            # show w opencv
            cv2.imshow("moring", img_show)
            cv2.imwrite("1.1.png", img_show)

            # loop delay
            cv2.waitKey(1)
    except KeyboardInterrupt:
        print("\nUser interrupt detected. Closing OpenCV...")
    finally:
        cv2.destroyAllWindows()
        print("Program Ended")