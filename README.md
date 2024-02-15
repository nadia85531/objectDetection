# objectDetection
Extraction of moving objects and detection of pedestrians from a sequence of images or video is often used 
in many video analysis tasks. For instance, it is a key component in intelligent video surveillance systems
and autonomous driving.

The program is developed in Python using OpenCV 4.6.0 to detect, separate and count moving objects from a given sequence of images or video 
captured by a stationary camera and to detect pedestrians. There are two parts:
1. Background modelling: extract moving objects using Gaussian Mixture background modelling
2. Detection and Tracking of Pedestrians: detect pedestrians using a OpenCV Deep Neural Network (DNN) module and 
a MobileNet SSD detector pre-trained on the MS COCO dataset, to track and display the detected pedestrians.
