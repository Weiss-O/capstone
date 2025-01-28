#This file has the code for setting up the device.
#The device needs to home and then take baseline images at each baseline position

import config
import Camera #TODO: Implement this Module
import Control #TODO: Implement this Module

#Initialize the controller
teensy = Control.Controller(config.controller_settings) #TODO: Implement this class
teensy.connect() #TODO: Implement this method
teensy.home() #TODO: Implement this Method

#Initialize the camera
camera = Camera.Camera(config.camera_settings) #TODO: Implement this class
camera.start() #TODO: Implement this method

#Find the required camera positions to cover the whole room
def find_required_positions():
    pass

for key in config.baseline:
    baseline_image_path = config.baseline[key]["image"]
    camera_pos = config.baseline[key]["camera_pos"]

    teensy.move_to(camera_pos)
    camera.capture_image_file(baseline_image_path)

camera.stop() #TODO: Implement this method
teensy.disconnect() #TODO: Implement this method