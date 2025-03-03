# This script will handle the high-level functionality on the raspberry pi.
# The various lower-level functionalities of the program will be imported as modules.
# This script will run on the raspberry pi

import yaml
with open('config.yaml') as file:
    config = yaml.safe_load(file)


#Import the necessary modules
import os
from abc import ABC, abstractmethod
from enum import Enum
import time
import cv2 #Currently used to read baseline image which is passed to baseline detector


if os.environ.get('RPI', 'False').lower()== 'true':
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
import PersonDetection
import ProposalGenerator as PG
import OPO

scanned_for_objects = False

#Enumeration class for the device states
class DeviceState(Enum):
    VACANT = 0
    OCCUPIED = 1

#Define device states
class State(ABC):
    @abstractmethod
    def handle():
        pass

#State for when the user is not in the room
class Occupied(State):
    @staticmethod
    def handle():
        global scanned_for_objects
        if(checkPointingConditions()):
            try: 
                point()
            except Exception as e:
                print(f"Error: {e}")
                teensy.home()
        PERSON_DETECTED, _ = scan()

        if os.environ.get('PD', 'Auto').lower()== 'manual':
            PD = input("Is there a person in the room? (Y/n)")
            PERSON_DETECTED = PD == "Y"

        if PERSON_DETECTED:
            print("Person Detected! Room Still Occupied")
            return DeviceState.OCCUPIED
        else:
            print("Scan complete without person detected.\nRoom now vacant.")
            idle(config["idle_time_vacant"])
            scanned_for_objects = False
            return DeviceState.VACANT

#State for when the user is in the room
#Dictionary to store the detector for each baseline
detectors={}
detectedObjects = []

class Vacant(State):
    @staticmethod
    def handle():
        global scanned_for_objects
        PERSON_DETECTED, imageArray = scan()
        if PERSON_DETECTED:
            print("Person Detected! State switched to Occupied.")
            idle(config["idle_time_occupied"])
            return DeviceState.OCCUPIED
        else:
            #TODO: Find a better way of managing the detected objects (clearing, adding to baseline, status, etc.)
            if not(scanned_for_objects):
                for image_prompt in imageArray:
                    detectObjects(image_prompt["image"], image_prompt["POSID"])
                scanned_for_objects = True
                
            idle(config["idle_time_vacant"])
            return DeviceState.VACANT


#Function to detect objects in an image
#It checks if a detector has been initialized for the position, if not, it initializes one
#TODO: It might be better to have the detector class handle the different baseline images itself
#      to avoid creating so many predictors in the main script (still same number of predictors would need to be created)
def detectObjects(image, POSID):
    if POSID not in detectors.keys():
        detectors[POSID] = Detector.RemoteDetector(POSID=POSID,
                                                   server=server)
    
    detections = detectors[POSID].detect(image)

    for detection in detections:
        detectedObjects.append(OPO.Object(detection, POSID))


def checkPointingConditions():
    return len(detectedObjects) > 0

def scan():
    image_array = []
    for POSID in config["baseline"].keys():
        #Move to position
        pos = config["baseline"][POSID]["camera_pos"]
        theta_actual, phi_actual = teensy.point_camera(pos[0], pos[1])
        time.sleep(1) #Wait for the camera to stabilize
        image = camera.capture()
        print (f"Captured image at position ({theta_actual}, {phi_actual})")
        if os.environ.get('IGNORE_PEOPLE', 'False').lower()== 'true':
            PERSON_DETECTED = False
        else:
            PERSON_DETECTED = PersonDetection.detect_person(image=image)
        if PERSON_DETECTED:
            teensy.home()
            return True, [] #Return empty list because no further processing should be done
        else:
            image_array.append({"image": image, "POSID": POSID})
    teensy.home()
    return False, image_array

def idle(t): #TODO: Look into whether this is the best thing to be doing in the idle state or whether there should be other things happening
    time.sleep(t)

def point(): #This is not the full functionality. The projection functionality needs to be implemented. For now we will point the camera.
    for obj in detectedObjects[:]:  # Create a slice copy of the list
        #TODO: There has to be a better way of doing this
        camera.update_ReferenceFrame(obj.camera_position[0], obj.camera_position[1]) #Update the camera theta_phi to be the ones used to capture the image containing the object
        pointing_ray = camera.calculate_pointing_ray(obj.center_coordinate, degrees=True)
        theta_actual, phi_actual = teensy.point_camera(pointing_ray[0], pointing_ray[1])
        print(f"Pointed camera to ({theta_actual}, {phi_actual})")
        time.sleep(1)
        if os.environ.get('SEND_OBJECT_PHOTOS', 'False').lower()== 'true':
                camera.capture_and_send_remote(server, f"object_capture-{obj.center_coordinate[0]}x{obj.center_coordinate[1]}y")

        time.sleep(0.5)
        detectedObjects.remove(obj)

#Main function
if __name__ == "__main__":    
    # -----------------------------------
    # Initialize important class instances
    # -----------------------------------

    #Initialize the camera
    cameraType = Camera.PiCamera if os.environ.get('RPI', 'False').lower() == 'true' else Camera.CameraStandIn #TODO: Make it so that this takes photo with  
    camera = cameraType(config["camera_settings"])
    camera.start()
    
    #Initialize the controller
    controllerType = Controller.Controller if os.environ.get('RPI', 'False').lower() == 'true' else Controller.ControllerStandIn
    teensy = controllerType(config["controller_settings"]) #TODO: Implement this class
    if not teensy.is_open:
        raise Exception("Controller not connected")
    time.sleep(2)

    # state = config['initial_state']
    state = DeviceState.VACANT
    while(True):
        if state == DeviceState.OCCUPIED:
            state = Occupied.handle()
        elif state == DeviceState.VACANT:
            state = Vacant.handle()
        else:
            raise Exception("Invalid State")

    
