# This script will handle the high-level functionality on the raspberry pi.
# The various lower-level functionalities of the program will be imported as modules.
# This script will run on the raspberry pi

#Import the necessary modules
import socket
import os
from abc import ABC, abstractmethod
from enum import Enum
import json


#Load Configuration File
import config

from Detector import BasicDetector

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

initial_state = DeviceState.OCCUPIED
            
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
    camera = Camera(config.camera_settings) #TODO: Implement this class
    #Check that the camera is 
    if camera.is_connected() == False:
        print("Camera is not connected")
        exit()
    
    #Initialize the controller
    teensy = Controller(config.controller_settings) #TODO: Implement this class

    if teensy.is_connected() == False:
        print("Controller is not connected")
        exit()

    #Check the server connection
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.connect((config.server_ip, config.server_port))
    #Send message to server
    if server is None:
        print("Server is not connected")
        exit()
    
    state = initial_state
    while(True):
        if state == DeviceState.OCCUPIED:
            state = Occupied.handle()
        elif state == DeviceState.VACANT:
            state = Vacant.handle()

    
