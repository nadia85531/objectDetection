Moving Object Detection and Analysis
This Python script is designed for detecting and analyzing moving objects within video files. It utilizes OpenCV for processing and analysis, focusing on pedestrian detection, object classification, and distance estimation based on the Y-coordinate in the video frame.

Features
Background Subtraction: Utilizes MOG2 method for background subtraction to isolate moving objects.
Object Detection: Detects objects and classifies them as persons, cars, or others based on their shape and aspect ratio.
Distance Estimation: Estimates the distance of detected pedestrians from the camera using the center Y-coordinate of bounding boxes.
Pedestrian Tracking: Tracks pedestrians across frames and labels them for identification.
Visual Display: Provides a visual display of the original video, detected objects, and the analysis results in a grid format.
Prerequisites
Python 3.x
OpenCV (cv2)
NumPy
Matplotlib (Optional for additional visualization)
Ensure all dependencies are installed using pip:

sh
Copy code
pip install opencv-python numpy matplotlib
Usage
The script offers two main functionalities through command-line arguments:

Task 1: Background subtraction and object detection/classification
Task 2: Pedestrian detection and tracking
To run the script, navigate to the directory containing movingObj.py and execute one of the following commands in the terminal.

For Task 1:
sh
Copy code
python movingObj.py -b <videofile>
For Task 2:
sh
Copy code
python movingObj.py -d <videofile>
Replace <videofile> with the path to your video file.

Output
The script processes the input video and displays a window with the video analysis results. For Task 1, it shows the original video, detected moving objects, and classified objects. For Task 2, it identifies and tracks pedestrians, highlighting the closest ones.

Press Esc key to exit the video display window.
