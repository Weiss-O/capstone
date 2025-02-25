from abc import ABC, abstractmethod
import yaml
import Detector

with open('config.yaml') as file:
    config = yaml.safe_load(file)

class Object():
    def __init__(self, detection:Detector.Detection, POSID):
        self.camera_position = config["baseline"][POSID]["camera_pos"] 
        self.bbox = detection.get_as_array() #Bounding box [x, y, w, h]
        self.center_coordinate = [self.bbox[0] + self.bbox[2]/2, self.bbox[1] + self.bbox[3]/2, 1] #Center coordinate [x, y, 1]
        #Can potentially add more attributes to store state, pointing ray, etc.

