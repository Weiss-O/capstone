import cv2
import os
import time
import sys

# Ensure the test_set directory exists

def capture_images(interval_minutes, total_duration_minutes, directory):
    # Convert minutes to seconds for easier calculations
    interval_seconds =  interval_minutes
    total_duration_seconds = total_duration_minutes * 60

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    start_time = time.time()
    elapsed_time = 0
    image_count = 1  # Start naming from 1

    while elapsed_time < total_duration_seconds:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Save the image with a sequential filename
        filename = f"/capture_{image_count}.jpg"
        cv2.imwrite(directory + filename, frame)
        print(f"Saved: {filename}")

        # Wait for the specified interval
        time.sleep(interval_seconds)
        elapsed_time = time.time() - start_time
        image_count += 1

    # Release the webcam
    cap.release()
    print(f"Captured {image_count - 1} images in total.")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python capture_images.py <interval_minutes> <total_duration_minutes>")
        sys.exit(1)

    interval_minutes = float(sys.argv[1])
    total_duration_minutes = float(sys.argv[2])
    directory = sys.argv[3]

    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        print("Error: Directory already exists.")
        sys.exit(1)
    

    capture_images(interval_minutes, total_duration_minutes, directory)
