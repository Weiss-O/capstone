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
import time
import cv2

if os.environ.get('RPI', 'False').lower == 'true':
    import socket
    HOST = config["server_settings"]["HOST"]
    PORT = config["server_settings"]["PORT"]
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.connect((HOST, PORT))

# import necessary modules
import Detector
import Camera
import Controller
import Classifier

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
            for image_prompt in imageArray:
                detections = detectObjects(image_prompt["image"], image_prompt["POSID"]) #TODO: Need to correlate the image with the position
                for detection in detections:
                    detectedObjects.append(OPO.Object(detection, image_prompt["POSID"])) #TODO: Only store the necessary information about the object, not all the detection information
            #TODO: Store detected objects in device memory

            idle(config["idle_time_vacant"])
            return DeviceState.VACANT

#Dictionary to store the detector for each baseline
detectors={}
detectedObjects = []

#Initialize the test predictor - this is used for every test image prompt
if os.environ.get('RPI', 'False').lower == 'true':
    testPredictor = Classifier.RemotePredictor(server)
else:
    testPredictor = Classifier.SAM2Predictor()


#Function to detect objects in an image
#It checks if a detector has been initialized for the position, if not, it initializes one
#TODO: It might be better to have the detector class handle the different baseline images itself
#      to avoid creating so many predictors.
def detectObjects(image, POSID):
    if POSID not in detectors.keys():
        if os.environ.get('RPI', 'False').lower == 'true':
            baselinePredictor = Classifier.RemotePredictor(server)
        else:
            baselinePredictor = Classifier.SAM2Predictor()

        baselineImage = cv2.imread(config["baseline"][POSID]["image_path"])

        baselineClassifier = Classifier.IOUSegmentationClassifier(baseline_predictor=baselinePredictor,
                                                                  test_predictor=testPredictor,
                                                                  iou_threshold=0.5,
                                                                  baseline_image=baselineImage)
        detectors[POSID] = Detector.BasicDetector(baseline=baselineImage,
                                                  classifier=baselineClassifier)
    return detectors[POSID].detect(image)


def checkPointingConditions():
    return False

def detectPerson(image=None):
    return False

def scan():
    image_array = []
    for POSID in config["baseline"].keys():
        image = camera.capture()
        PERSON_DETECTED = detectPerson(image=None) #TODO: Implement person detection
        if PERSON_DETECTED:
            return True, [] #Return empty list because no further processing should be done
        else:
            image_array.append({"image": image, "POSID": POSID})
    return False, image_array

def idle(t):
    time.sleep(t)

def point(): #This is not the full functionality. The projection functionality needs to be implemented. For now we will point the camera.
    for obj in detectedObjects:
        teensy.point_camera(obj.camera_position)
        time.sleep(1)
        
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

    
