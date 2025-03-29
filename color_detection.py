"""
File Name: color_detection.py
Author: Yutao He
Date Created: 2025-03-28
Description: 
    Different color detection algorithms
    - RGB Thresholding
    - HSV Thresholding
    - Lab Color Space Detection
    - YUV Color Space Detection
"""
import cv2
import numpy as np
import csv
import os
from datetime import datetime

# Directory for saving experiment data
DATA_DIR = "experiment_data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# CSV file for saving data
CSV_FILE = os.path.join(DATA_DIR, "color_detection_data.csv")

# Create CSV file with headers if it doesn't exist
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Color", "Lighting Conditions", "Algorithm", "Detected Pixels"])

# HSV Ranges for the color red
HSV_RANGES = {"red": ((0, 120, 70), (10, 255, 255))}

# RGB Thresholds for the color red
RGB_RANGES = {"red": ((0, 0, 128), (100, 100, 255))}

# Function: RGB Thresholding
def detect_rgb(frame, color_name):
    lower, upper = RGB_RANGES[color_name]
    mask = cv2.inRange(frame, np.array(lower), np.array(upper))
    detected_pixels = cv2.countNonZero(mask)
    return mask, detected_pixels

# Function: HSV Thresholding
def detect_hsv(frame, color_name):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower, upper = HSV_RANGES[color_name]
    mask = cv2.inRange(hsv_frame, np.array(lower), np.array(upper))
    detected_pixels = cv2.countNonZero(mask)
    return mask, detected_pixels

# Function: Lab Color Space Detection
def detect_lab(frame, color_name):
    lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
    # Example thresholds for red (adjust as needed)
    lower = np.array([20, 150, 150])
    upper = np.array([255, 200, 200])
    mask = cv2.inRange(lab_frame, lower, upper)
    detected_pixels = cv2.countNonZero(mask)
    return mask, detected_pixels

# Function: YUV Color Space Detection
def detect_yuv(frame, color_name):
    yuv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    # Example thresholds for red
    lower = np.array([0, 120, 140])
    upper = np.array([255, 180, 200])
    mask = cv2.inRange(yuv_frame, lower, upper)
    detected_pixels = cv2.countNonZero(mask)
    return mask, detected_pixels

# Save Data to CSV
def record_data(timestamp, color, lighting_conditions, algorithm, detected_pixels):
    with open(CSV_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, color, lighting_conditions, algorithm, detected_pixels])

# Main Function
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 's' to save data, 'q' to quit.")
    algorithms = {
        "RGB": detect_rgb,
        "HSV": detect_hsv,
        "Lab": detect_lab,
        "YUV": detect_yuv,
    }

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Resize for faster processing
        frame = cv2.resize(frame, (640, 480))

        # Choose Algorithm
        print("Available Algorithms: HSV, RGB, Lab, Histogram, YUV, Mean Shift, Adaptive")
        algorithm_name = input("Enter the algorithm to use: ")
        color_name = "red"  # Change to desired color

        if algorithm_name not in algorithms:
            print("Invalid algorithm. Try again.")
            continue

        # Run Selected Algorithm
        mask, detected_pixels = algorithms[algorithm_name](frame, color_name)

        # Show Results
        cv2.imshow("Original", frame)
        cv2.imshow(f"{algorithm_name} Detection", mask)

        # Save data
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            lighting_conditions = input("Enter lighting conditions (e.g., 'Bright', 'Dim'): ")
            record_data(timestamp, color_name, lighting_conditions, algorithm_name, detected_pixels)
            print(f"Data saved: {timestamp}, {color_name}, {lighting_conditions}, {algorithm_name}, {detected_pixels} pixels")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
