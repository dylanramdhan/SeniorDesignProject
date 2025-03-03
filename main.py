"""
    ### Senior Design Project ###
    Engineer: Dylan Ramdhan
    
    Tools: YOLOv8, PyTorch, OpenCV, and Ultralytics
    Task: Detect and crop nutrition labels from a webcam feed
    
    This script will take a picture from the webcam and crop the image to the most significant region.
    This is useful for cropping out labels from products.
"""

## Note to self: Feed more images of nutrition labels to the model to improve detection accuracy

import cv2
import torch
import os
from ultralytics import YOLO


def setup():
    """Ensure the save directory exists."""
    save_folder = "cropped-labels"
    os.makedirs(save_folder, exist_ok=True)
    return save_folder


def detect_and_crop_labels(reference_image_path, save_folder):
    # detecting and cropping labels based on refernce images using YOLOv8
    # load YOLOv8 model (using a larger model for better detection)
    model = YOLO("runs/detect/train3/weights/best.pt")  # using the trained model
    
    # Open webcam first to ensure it's available
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    print("Press 's' to capture and process, or 'q' to quit.")
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to grab frame")
            break
        
        cv2.imshow("Camera Feed", frame)
        key = cv2.waitKey(1) & 0xFF
        
        
        if key == ord('s'):  # Capture image and process
            cap.release()
            cv2.destroyAllWindows()
            
            # Perform YOLO detection on captured frame
            results = model(frame)
            objects = results[0].boxes.xyxy
            
            if len(objects) == 0:
                print("No labels detected in camera feed.")
            else:
                for i, (x1, y1, x2, y2) in enumerate(objects):
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    cropped = frame[y1:y2, x1:x2]
                    save_path = os.path.join(save_folder, f"cropped_label_{i}.jpg")
                    cv2.imwrite(save_path, cropped)
                    print(f"Cropped label saved at: {save_path}")
            return
        
        
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return


# Usage
save_folder = setup()
detect_and_crop_labels("nutrition_label.png", save_folder)










######  ONLY takes one photo... not quite accurate#######
# import cv2
# import torch
# import os
# import numpy as np
# from ultralytics import YOLO

# def setup():
#     """Ensure the save directory exists."""
#     save_folder = "cropped-labels"
#     os.makedirs(save_folder, exist_ok=True)
#     return save_folder

# def feature_matching(img1, img2):
#     """Use ORB feature matching to compare detected object with reference image."""
#     orb = cv2.ORB_create()
#     keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
#     keypoints2, descriptors2 = orb.detectAndCompute(img2, None)
    
#     bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#     matches = bf.match(descriptors1, descriptors2)
    
#     return len(matches)  # Higher match count = more similarity

# def detect_and_crop_labels(reference_image_path, save_folder):
#     """Detect and crop labels based on a reference image using YOLOv8 and feature matching."""
#     model = YOLO("yolov8x.pt")
#     reference_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)
    
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("Error: Could not open webcam.")
#         return
    
#     print("Press 's' to capture and process, or 'q' to quit.")
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to grab frame")
#             break
        
#         cv2.imshow("Camera Feed", frame)
#         key = cv2.waitKey(1) & 0xFF
        
#         if key == ord('s'):  # Capture image and process
#             cap.release()
#             cv2.destroyAllWindows()
#             results = model(frame)
#             objects = results[0].boxes.xyxy
            
#             if len(objects) == 0:
#                 print("No labels detected in camera feed.")
#             else:
#                 best_match = None
#                 highest_score = 0
                
#                 for i, (x1, y1, x2, y2) in enumerate(objects):
#                     x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
#                     detected_label = frame[y1:y2, x1:x2]
#                     detected_label_gray = cv2.cvtColor(detected_label, cv2.COLOR_BGR2GRAY)
#                     match_score = feature_matching(reference_image, detected_label_gray)
                    
#                     if match_score > highest_score:
#                         highest_score = match_score
#                         best_match = (x1, y1, x2, y2)
                
#                 if best_match:
#                     x1, y1, x2, y2 = best_match
#                     cropped = frame[y1:y2, x1:x2]
#                     save_path = os.path.join(save_folder, "cropped_label.jpg")
#                     cv2.imwrite(save_path, cropped)
#                     print(f"Cropped label saved at: {save_path}")
#                 else:
#                     print("No good match found for the reference label.")
#             return
        
#         elif key == ord('q'):
#             cap.release()
#             cv2.destroyAllWindows()
#             return

# # Usage
# save_folder = setup()
# detect_and_crop_labels("nutrition_label.png", save_folder)