import socket
from picamera2 import Picamera2
from libcamera import controls
import time

# Initialize the camera
picam2 = Picamera2()

# Configure camera for high resolution
camera_config = picam2.create_still_configuration(main={"size": (4608 , 2592)})  # Max resolution for Pi Camera v2
picam2.configure(camera_config)

# Set the manual focus mode
picam2.set_controls({"AfMode": controls.AfModeEnum.Manual, "LensPosition": 0})  # Adjust LensPosition as needed
picam2.start(show_preview=False)

# Allow the camera to settle
time.sleep(1)

# Capture the image
image_path = "/home/weiso6959/captured_image.jpg"  # Path to save the image
picam2.capture_file(image_path)
picam2.stop()


# Socket setup
HOST = '100.72.50.30'  # Tailscale IP of your computer
PORT = 5000

# Send the image file to the server
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
    client.connect((HOST, PORT))
    print("Connected to server")

    # Open the image file and send its contents
    with open(image_path, "rb") as image_file:
        while True:
            chunk = image_file.read(1024)  # Read in chunks of 1KB
            if not chunk:
                break
            client.sendall(chunk)  # Send each chunk to the server

    print("Image sent successfully")
