#This file has the code for setting up the device.
#The device needs to home and then take baseline images at each baseline position

from Camera import PiCamera #TODO: Implement this Module
from Controller import Controller #TODO: Implement this Module
import time
import yaml
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


#Initialize the camera
camera = PiCamera(config["camera_settings"]) #TODO: Implement this class
camera.start() #TODO: Implement this method


nominal_baseline_positions = [[45, 60], [80, 39], [45, 39], [10, 39]]

#TODO: Implement the below function properly
#Find the required camera positions to cover the whole room
def find_required_positions():
    return nominal_baseline_positions


teensy.home() #TODO: Implement this Method
points = find_required_positions()

#TODO: The below loop will assign points but it will not 
#Move to each baseline position and take an image
for i, point in enumerate(points):
    theta_actual, phi_actual = teensy.point_camera(point[0], -point[1]) #TODO: Fix the Arduino so that the positive direction corresponds correctly to the positive phi axis
    time.sleep(1) #Wait for the camera to stabilize
    
    image_path = os.path.join(os.path.abspath(__file__), f"baseline/baseline{i}.jpg")
    camera.capture_file(image_path)
    time.sleep(2)

    config["baseline"][f"POS{i}"]["image_path"] = image_path
    config["baseline"][f"POS{i}"]["camera_pos"] = [theta_actual, phi_actual]

camera.stop() #TODO: Implement this method

with open("config.yaml", "w") as file:
    yaml.dump(config, file)