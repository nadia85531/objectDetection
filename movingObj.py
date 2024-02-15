import argparse
import cv2 as cv
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

'''
For each detected pedestrian bounding box, first calculate its center's Y-coordinate. 
The Y-coordinate can be used as an estimated indicator to judge 
the distance of objects from camera; the larger the Y-coordinate, the closer the pedestrian is.
Use a dictionary to store each pedestrian bounding box and its corresponding center Y-coordinate.
Next, sort these bounding boxes based on the center Y-coordinates.
'''


def main():
    args = parse_and_run()

    if args.b:  
        video_filename = args.videofile
        
        if not os.path.isfile(video_filename):
            print(f"File not found: {video_filename}")
            sys.exit(1)
            
        task1(video_filename)
        
    elif args.d:  
        video_filename = args.videofile
        
        if not os.path.isfile(video_filename):
            print(f"File not found: {video_filename}")
            sys.exit(1)
            
        task2(video_filename)
        
    else:
        print("Invalid arguments. Usage: python a3.py -b videofile OR python a3.py -d videofile")
        sys.exit(1)


def task1(video_filename):

    # Open the video file
    cap = cv.VideoCapture(video_filename)
    
    # Check if the video opened successfully
    if not cap.isOpened():
        print("Error: Unable to open the video file.")
        sys.exit(1)

    # Get the frame rate of the video
    fps = cap.get(cv.CAP_PROP_FPS)
    delay = int(1000/fps)

    # Create a background subtractor using MOG2
    bg_subtractor = cv.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

    def process_fg_mask(fg_mask):
    
        # Increase the size of the kernel
        kernel_opening = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
        kernel_closing = cv.getStructuringElement(cv.MORPH_ELLIPSE, (9, 9))
    
        # Use the opening operation to remove noise
        opening = cv.morphologyEx(fg_mask, cv.MORPH_OPEN, kernel_opening)
    
        # Use the closing operation to enhance the contours of objects
        closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel_closing)

        # Thresholding operation to binarize and remove shadows
        _, bin_mask = cv.threshold(closing, 127, 255, cv.THRESH_BINARY)
    
        # Further reduce noise using the erosion operation
        eroded_mask = cv.erode(bin_mask, kernel_opening, iterations=1)
    
        # Use dilation operation to restore the shape of the object
        dilated_mask = cv.dilate(eroded_mask, kernel_opening, iterations=2)
    
        # Filter out small regions
        contours, _ = cv.findContours(dilated_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv.contourArea(contour) < 300:  # Adjusted threshold
                cv.drawContours(dilated_mask, [contour], 0, 0, -1)
    
        return dilated_mask

    # Classify the object based on its width-height ratio
    def classify_object(w, h):
        ratio = w/h
        if 0.4 < ratio < 0.8:
            return "person"
        elif 1.2 < ratio < 2.5:
            return "car"
        else:
            return "other"
    
    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        # If the frame read was unsuccessful (end of video), break out of the loop
        if not ret:
            break

        # Resize the frame to VGA size 
        height, width, _ = frame.shape
        new_width = int(height * 4 / 3)  # Calculate new width based on its height

        if new_width > width:  # If new width is bigger than original width, reproduce the height based on the width
            new_width = width
            new_height = int(width * 3 / 4)
        else:
            new_height = height

        original_frame = cv.resize(frame, (new_width, new_height))
        
        
        # Apply the original_frame to the background subtractor
        fg_mask = bg_subtractor.apply(original_frame)
        
        # Process the foreground mask to reduce noise
        processed_fg_mask = process_fg_mask(fg_mask)
        
        # Set the detected_moving_pixels_frame to the foreground mask
        detected_moving_pixels_frame = fg_mask

        # Retrieve estimated background image from the Gaussian Mixture Model background subtractor
        estimated_background_frame = bg_subtractor.getBackgroundImage()
        
        # Convert the processed foreground mask to a 3-channel mask
        three_channel_mask = cv.cvtColor(processed_fg_mask, cv.COLOR_GRAY2BGR)
      
        # Use the 3-channel mask to extract the colored moving objects from the original frame
        detected_objects_frame = cv.bitwise_and(original_frame, three_channel_mask)
        
        # Connected Component Analysis
        retval, labels = cv.connectedComponents(processed_fg_mask)
        
        # Count objects by type
        object_count = {"person": 0, "car": 0, "other": 0}
        
        # Analyze labeled components
        for label_num in range(1, retval):  # Start from 1 to skip background
            
            # Create a mask for the current component
            component_mask = np.where(labels == label_num, 255, 0).astype('uint8')
            
            # Find bounding box for the component
            x, y, w, h = cv.boundingRect(component_mask)
            
            # Classify object
            object_type = classify_object(w, h)
            object_count[object_type] += 1
            

        # Print object counts
        total_objects = sum(object_count.values())
        if total_objects == 0:
            print(f"Frame {int(cap.get(cv.CAP_PROP_POS_FRAMES))}: 0 objects")
        else:
            print(f"Frame {int(cap.get(cv.CAP_PROP_POS_FRAMES))}: {total_objects} objects "
                  f"({object_count['person']} persons, {object_count['car']} cars, {object_count['other']} others)")

        display_grid(original_frame, estimated_background_frame, detected_moving_pixels_frame, detected_objects_frame, delay)
        
    # Release the video capture object after processing
    cap.release()


def display_grid(original_frame, estimated_background_frame, detected_moving_pixels_frame, detected_objects_frame, delay):
    MAX_WIDTH = 960
    MAX_HEIGHT = 720

    # Resize the frame to exactly half of the canvas size
    def resize_to_fit(frame):
        # Quadrant size
        QUAD_WIDTH = MAX_WIDTH // 2
        QUAD_HEIGHT = MAX_HEIGHT // 2
        
        if len(frame.shape) == 2:  # Check if it's a grayscale image
            frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
            
        return cv.resize(frame, (QUAD_WIDTH, QUAD_HEIGHT))

    # Apply resize_to_fit() function
    original_frame = resize_to_fit(original_frame)
    estimated_background_frame = resize_to_fit(estimated_background_frame)
    detected_moving_pixels_frame = resize_to_fit(detected_moving_pixels_frame)
    detected_objects_frame = resize_to_fit(detected_objects_frame)

    # Create an empty canvas
    canvas = np.zeros((MAX_HEIGHT, MAX_WIDTH, 3), dtype=np.uint8)

    # Place each resized frame on the canvas
    canvas[0:MAX_HEIGHT//2, 0:MAX_WIDTH//2] = original_frame
    canvas[0:MAX_HEIGHT//2, MAX_WIDTH//2:] = estimated_background_frame
    canvas[MAX_HEIGHT//2:, 0:MAX_WIDTH//2] = detected_moving_pixels_frame
    canvas[MAX_HEIGHT//2:, MAX_WIDTH//2:] = detected_objects_frame  

    # Display the canvas
    cv.imshow('', canvas)
    
    # Check for user input
    key = cv.waitKey(delay) & 0xFF  # Get the ASCII value of the key pressed
    
    # Press 'Esc' to exit the loop
    if key == 27:
        cv.destroyAllWindows()
        sys.exit(0)

def task2(video_filename):

    # For unique labels on pedestrians
    current_label = 1
    pedestrian_labels = dict()  # To store bounding boxes with their unique labels
    pedestrian_centerY = dict() # Store the Y-coordinate of the center of bounding boxes

    # Load the MobileNet SSD model and configuration using OpenCV's DNN module
    net = cv.dnn.readNetFromTensorflow('frozen_inference_graph.pb', 'ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt')

    # Open the video file
    cap = cv.VideoCapture(video_filename)
    
    # Check if the video opened successfully
    if not cap.isOpened():
        print("Error: Unable to open the video file.")
        sys.exit(1)

    # Get the frame rate of the video
    fps = cap.get(cv.CAP_PROP_FPS)
    delay = int(1000/fps)

    while True:
        
        pedestrian_labels = dict()
        
        pedestrian_centerY = dict()  # Reset centerY dictionary for each frame
        
        # Read a frame from the video
        ret, frame = cap.read()

        # If the frame read was unsuccessful (end of video), break out of the loop
        if not ret:
            break

        # Resize the frame to VGA size 
        height, width, _ = frame.shape
        new_width = int(height * 4 / 3)  # Calculate new width based on its height

        if new_width > width:  # If new width is bigger than original width, reproduce the height based on the width
            new_width = width
            new_height = int(width * 3 / 4)
        else:
            new_height = height

        original_frame = cv.resize(frame, (new_width, new_height))
        
        # Use the DNN module to perform pedestrian detection
        blob = cv.dnn.blobFromImage(original_frame, 1.0, (300, 300), (127.5, 127.5, 127.5), swapRB=True, crop=False)
        net.setInput(blob)
        detections = net.forward()

        # Create a copy of the resized frame for detected pedestrians
        detected_pedestrians = original_frame.copy()
        
        # Create a copy of the resized frame for detected pedestrians
        labelled_pedestrians = original_frame.copy()


        # Draw bounding boxes for detected pedestrians
        h, w = original_frame.shape[:2]
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.2:
                class_id = int(detections[0, 0, i, 1])
                if class_id == 1:  # ID for "person" in COCO dataset
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype('int')
                    
                    # Check the aspect ratio of the bounding box
                    bbox_width = endX - startX
                    bbox_height = endY - startY
                    aspect_ratio = bbox_height / bbox_width
                    
                    # If aspect ratio is less than a threshold (e.g., 1.5), skip this bounding box
                    if aspect_ratio < 1.5:
                        continue
                    
                   
                    # Check if this bounding box is already labeled
                    existing_label = None
                    for existing_bbox, lbl in pedestrian_labels.items():
                        ex_startX, ex_startY, ex_endX, ex_endY = existing_bbox
                        if abs(ex_startX - startX) < 10 and abs(ex_startY - startY) < 10 and abs(ex_endX - endX) < 10 and abs(ex_endY - endY) < 10:
                            existing_label = lbl
                            break
                        
                    # If not already labeled, label it and store
                    if existing_label is None:
                        pedestrian_labels[(startX, startY, endX, endY)] = current_label
                        label_text = "Person " + str(current_label)
                        current_label += 1
                        
                    else:
                        label_text = "Person " + str(existing_label)
                    
                    # Draw rectangle and label
                    cv.rectangle(detected_pedestrians, (startX, startY), (endX, endY), (0, 0, 255), 1)
                    cv.rectangle(labelled_pedestrians, (startX, startY), (endX, endY), (255, 0, 0), 1)
                    cv.putText(labelled_pedestrians, label_text, (startX, startY - 10), cv.FONT_HERSHEY_DUPLEX, 1, (255,0,0), 2)

                    # Calculate the Y-coordinate of the center of this bounding box
                    centerY = (startY + endY) / 2
                    pedestrian_centerY[(startX, startY, endX, endY)] = centerY

        # Find the three closest pedestrians (using center Y-coordinates)
        num_detected = len(pedestrian_centerY)
        if num_detected <= 3:
            closest_persons = sorted(pedestrian_centerY.items(), key=lambda x: x[1], reverse=True)[:num_detected]
        else:
            closest_persons = sorted(pedestrian_centerY.items(), key=lambda x: x[1], reverse=True)[:3]

        # Create a copy of the resized frame for the closest pedestrians
        closest_pedestrians = original_frame.copy()
        
        # Draw bounding boxes only for the three closest pedestrians
        for bbox, _ in closest_persons:
            startX, startY, endX, endY = bbox
            cv.rectangle(closest_pedestrians, (startX, startY), (endX, endY), (0, 255, 0), 3)

        display_frame(original_frame, detected_pedestrians, labelled_pedestrians, closest_pedestrians, closest_persons, delay)
        
    # Release the video capture object after processing
    cap.release()
    
        
def display_frame(original_frame, detected_pedestrians, labelled_pedestrians, closest_pedestrians, closest_persons, delay):
    
    MAX_WIDTH = 960
    MAX_HEIGHT = 720

    # Resize the frame to fit half of the canvas size
    def resize_to_fit(frame):
        # Quadrant size
        QUAD_WIDTH = MAX_WIDTH // 2
        QUAD_HEIGHT = MAX_HEIGHT // 2
        
        if len(frame.shape) == 2:  # Check if it's a grayscale image
            frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
            
        return cv.resize(frame, (QUAD_WIDTH, QUAD_HEIGHT))
    
    # Apply resize_to_fit() function
    original_frame = resize_to_fit(original_frame)
    detected_pedestrians = resize_to_fit(detected_pedestrians)
    labelled_pedestrians = resize_to_fit(labelled_pedestrians)
    closest_pedestrians_resized = resize_to_fit(closest_pedestrians) 
    
    # Create an empty canvas
    canvas = np.zeros((MAX_HEIGHT, MAX_WIDTH, 3), dtype=np.uint8)

    # Place each resized frame on the canvas
    canvas[0:MAX_HEIGHT//2, 0:MAX_WIDTH//2] = original_frame
    canvas[0:MAX_HEIGHT//2, MAX_WIDTH//2:] = detected_pedestrians
    canvas[MAX_HEIGHT//2:, 0:MAX_WIDTH//2] = labelled_pedestrians
    
    # On the last quadrant, highlight the three closest pedestrians
    canvas[MAX_HEIGHT//2:, MAX_WIDTH//2:] = closest_pedestrians_resized
    for bbox, _ in closest_persons:
        startX, startY, endX, endY = bbox
        cv.rectangle(canvas[MAX_HEIGHT//2:, MAX_WIDTH//2:], (startX, startY), (endX, endY), (0, 255, 0), 2)  

    # Display the canvas
    cv.imshow('', canvas)
    
    # Check for user input
    key = cv.waitKey(delay) & 0xFF  # Get the ASCII value of the key pressed
    
    # Press 'Esc' to exit the loop
    if key == 27:
        cv.destroyAllWindows()
        sys.exit(0)

def parse_and_run():
    parser = argparse.ArgumentParser(description='Process a video file for task 1 or task 2.')

    # Define the -b and -d as optional arguments
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-b', action='store_true', help='Option for task 1')
    group.add_argument('-d', action='store_true', help='Option for task 2')

    # Define the video file as a positional argument
    parser.add_argument('videofile', help='Video file for the selected task')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
