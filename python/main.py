# This script will handle the high-level functionality on the raspberry pi.
# The various lower-level functionalities of the program will be imported as modules.
# This script will run on the raspberry pi

import yaml
with open('config.yaml') as file:
    config = yaml.safe_load(file)
    print(config)


#Import the necessary modules
import socket
import os
from abc import ABC, abstractmethod
from enum import Enum
import json

# import config
from Detector import BasicDetector
import Camera
import Controller

#Enumeration class for the device states
class DeviceState(Enum):
    VACANT = 0
    OCCUPIED = 1

#Define device states
class State(ABC):
    @abstractmethod
    def handle(self):
        pass

#State for when the user is not in the room
class Vacant(State):
    def handle(self):
        PERSON_DETECTED, _ = scan()
        if PERSON_DETECTED:
            return DeviceState.OCCUPIED
        else:
            idle(config["idle_time_vacant"])
            return DeviceState.VACANT

#State for when the user is in the room
class Occupied(State):
    def handle(self):
        if(checkPointingConditions()):
                point()
        PERSON_DETECTED, imageArray = scan()
        if PERSON_DETECTED:
            idle(config["idle_time_occupied"])
            return DeviceState.OCCUPIED
        else:
            detectedObjects = []
            for image in imageArray:
                detections = detectObjects(image)
                detectedObjects.append(detections)
            #TODO: Store detected objects in device memory
            

            idle(config.idle_time_vacant)
            return DeviceState.VACANT
            
def checkPointingConditions():
    return False

def scan():
    return None, None

def idle():
    pass

def point():
    pass

#Main function
if __name__ == "__main__":    
    # -----------------------------------
    # Initialize important class instances
    # -----------------------------------

    #Initialize the camera
    cameraType = Camera.PiCamera if os.environ.get('RPI') == True else Camera.CameraStandIn
    camera = cameraType(config["camera_settings"])
    camera.start()
    
    #Initialize the controller
    teensy = Controller(config.controller_settings) #TODO: Implement this class

    try:
        teensy.connect()
    except Exception as e:
        print("Controller is not connected: ", e)
        exit()

    #Check the server connection
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.connect((config.server_ip, config.server_port))
    #Send message to server
    if server is None:
        print("Server is not connected")
        exit()
    
    state = config.initial_state
    while(True):
        if state == DeviceState.OCCUPIED:
            state = Occupied.handle()
        elif state == DeviceState.VACANT:
            state = Vacant.handle()

    
