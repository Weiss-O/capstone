#This file has the code for setting up the device.
#The device needs to home and then take baseline images at each baseline position

from Camera import PiCamera #TODO: Implement this Module
from Controller import Controller #TODO: Implement this Module
import time
import yaml
import datetime
import os

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

new_baseline = input("Are you sure you want to overwrite the baseline images? (Y/n)")

if new_baseline != "Y":
    exit()

config["baseline"] = {}
#Initialize the controller
teensy = Controller(config["controller_settings"])
if not teensy.is_open:
    raise Exception("Controller not connected")
time.sleep(5)

#Initialize the camera
camera = PiCamera(config["camera_settings"])
camera.start()


nominal_baseline_positions = [[0, 70], [15, 38.4], [15-68.7, 38.4], [0, 18.4]] #TODO: Re-calculate these positions for a horizontal camera

#TODO: Implement the below function properly
#Find the required camera positions to cover the whole room
def find_required_positions():
    return nominal_baseline_positions


teensy.home() #TODO: Implement this Method
points = find_required_positions()


#Get the current time to use as a unique identifier
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

#TODO: The below loop will assign points but it will not
#Move to each baseline position and take an image
for i, point in enumerate(points):
    theta_actual, phi_actual = teensy.point_camera(point[0], point[1]) #TODO: Fix the Arduino so that the positiv
    time.sleep(1) #Wait for the camera to stabilize

    baseline_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline")
    if not os.path.exists(baseline_dir):
        os.makedirs(baseline_dir)

    image_path = os.path.join(baseline_dir, f"baseline{i}_{current_time}.jpg")
    camera.capture_file(image_path)
    teensy.motors_off()
    time.sleep(2)

    config["baseline"][f"POS{i}"] = {"image_path": image_path, "camera_pos": [float(theta_actual), float(phi_actual)]}

teensy.home()
camera.stop() #TODO: Implement this method

with open("config.yaml", "w") as file:
    yaml.dump(config, file)