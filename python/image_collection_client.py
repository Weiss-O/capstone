#Code to collect images to test alignment

from Camera import PiCamera #TODO: Implement this Module
from Controller import Controller #TODO: Implement this Module
import time
import yaml
import os
import numpy as  np
import socket
import Server
import cv2

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

#Establish a connection to the server
HOST = config["server_settings"]["HOST"]
PORT = config["server_settings"]["PORT"]
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))

#Initialize the controller
teensy = Controller(config["controller_settings"])
if not teensy.is_open:
    raise Exception("Controller not connected")
time.sleep(2)

#Initialize the camera
camera = PiCamera(config["camera_settings"])
camera.start()

deg_per_step = 360/2048

nominal_baseline_positions = [[0, 70], [15, 38.4], [0, 18.4]]
offset_steps = np.array([2, 5, 10, 20])
offset_angles = deg_per_step*offset_steps
pan_tilt_combinations = [[0, 1], [1, 0], [1, 1], [-1, 0], [0, -1], [-1, -1], [1, -1], [-1, 1]]

def capture_and_send_image():
    image = camera.capture()
    success, encoded_image = cv2.imencode('.jpg', image)
    if not success:
        raise Exception("Error encoding image")
    image_bytes = encoded_image.tobytes()

    Server.send_bytes(s, image_bytes)

count = 0
try:
    for point in nominal_baseline_positions:
        #Move camera
        teensy.point_camera(point[0], point[1])
        time.sleep(1)

        #Capture image at nominal position
        capture_and_send_image()

        image_title = f"NP{point[0]:.2f}T{point[1]:.2f}_OP{0}T{0}"
        #Convert to bytes
        image_title_bytes = image_title.encode()
        
        #Send the title to the server
        Server.send_bytes(s, image_title_bytes)

        #Get response
        response = Server.get_response(s)
        if response != b"ACK":
            raise Exception(f"Expected ACK but got {response}")

        count += 1
        print(f"{count*100/123:.1f}%: Sent image: {image_title}")

        for offsets, steps in zip(offset_angles, offset_steps):
            for pt in pan_tilt_combinations:
                theta = point[0] + pt[0]*offsets
                phi = point[1] + pt[1]*offsets
                #Move camera to position
                teensy.point_camera(theta, phi)
                time.sleep(1)

                #Capture image at nominal position
                capture_and_send_image()

                #Generate image title
                image_title = f"NP{point[0]:.2f}T{point[1]:.2f}_OP{pt[0]*steps}T{pt[1]*steps}"
                #Convert to bytes
                image_title_bytes = image_title.encode()
                
                #Send the title to the server
                Server.send_bytes(s, image_title_bytes)
                #Get response
                response = Server.get_response(s)
                if response != b"ACK":
                    raise Exception(f"Expected ACK but got {response}")

                count += 1
                print(f"{count*100/123:.1f}%: Sent image: {image_title}")

    print(f"Total images captured: {count}")
except Exception as e:
    teensy.home()
    camera.stop()
    print(f"Error: {e}")