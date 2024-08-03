import cv2
import time
import logging
import os

def capture_images(framerate: int = ..., duration: float = ...):
    image_array = []

    # Create a VideoCapture object to access the camera
    cap = cv2.VideoCapture(1)  # 0 represents the default camera

    # Check if the camera is opened successfully
    if not cap.isOpened():
        logging.error("Failed to open the USB camera")
        exit()

    start_time = time.time()
    time_of_last_frame = time.time()
    # Read and display frames from the camera at a set interval
    while time.time() - start_time <= duration:
        # Wait for a key event or a specific time interval
        if cv2.waitKey(int(1000/framerate)) & 0xFF == ord('q'):
            break

        # Read a frame from the camera
        ret, frame = cap.read()

        # Check if the frame was successfully read
        if not ret:
            logging.error("Failed to read frame from the camera")
            image_array.append(None)
        else:
            # Add the frame to an array of images
            image_array.append(frame)
    # Release the VideoCapture object and close the window
    cap.release()
    cv2.destroyAllWindows()
    return image_array

#Function takes image array and displays as video looped with given framerate
def display_images(image_array: list, framerate: int = ...):
    # Display the frames in a loop
    counter = 0
    time_of_last_frame = time.time()
    while counter < len(image_array):
        image = image_array[counter]
        if image is not None:
            if time.time() - time_of_last_frame > 1/framerate:
                time_of_last_frame = time.time()
                cv2.imshow("Captured Frames", image)
                counter += 1
                counter = counter % len(image_array) #restart at end of loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def video_stream():
    # Create a VideoCapture object to access the camera
    cap = cv2.VideoCapture(1)  # 0 represents the default camera

    # Check if the camera is opened successfully
    if not cap.isOpened():
        logging.error("Failed to open the USB camera")
        exit()

    # Read and display frames from the camera
    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        # Check if the frame was successfully read
        if not ret:
            logging.error("Failed to read frame from the camera")
            break

        # Display the frame in a window
        cv2.imshow("Video Stream", frame)

        # Break the loop when the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the VideoCapture object and close the window
    cap.release()
    cv2.destroyAllWindows()

def save_images(image_array: list, folder_path: str):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    for i, image in enumerate(image_array):
        if image is not None:
            file_path = folder_path + f"/image_{i}.jpg"
            success = cv2.imwrite(file_path, image)
            if not success:
                print(f"Failed to save image {i} to {file_path}")

#prompt user for framerate and duration
framerate = float(input("Enter framerate: "))
duration = int(input("Enter duration in seconds: "))
num_images = int(duration*framerate)

#Allow user to position camera before capturing images
video_stream()

#Capture images
print(f"Capturing {num_images} images at {framerate} fps for {duration} seconds")
images = capture_images(framerate, duration)

#Save images to folder
directory = f'usb_camera_images\\{str(time.localtime().tm_year)}.{str(time.localtime().tm_mon)}.{str(time.localtime().tm_mday)}.{str(time.localtime().tm_hour)}.{str(time.localtime().tm_min)}.{str(time.localtime().tm_sec)}'
save_images(images, directory)

#Display images in a loop
display_images(images, 3)


# Release the VideoCapture object and close the window
cv2.destroyAllWindows()