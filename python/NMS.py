from abc import ABC, abstractmethod
import ContourAnalysis as CA
import numpy as np

class Merger(ABC):
    @abstractmethod
    def merge(self, detections) -> list:
        pass

class TestMaskIOUMerger(Merger):
    @staticmethod
    def merge(detections, iou_threshold) -> list:
        #Create a list to store the merged detections
        merged_detections = []
        #Iterate through the detections
        for detection in detections:
            #Set the flag to determine if the detection has been merged
            merged = False
            #Iterate through the merged detections
            for merged_detection in merged_detections:
                #Calculate the intersection over union of the detection and merged detection
                iou = CA.calculate_iou(detection.testMask, merged_detection.testMask)
                #If the intersection over union is greater than the threshold
                if iou > iou_threshold:
                    #Merge the detection into the merged detection
                    merged_detection.testMask = np.logical_or(detection.testMask, merged_detection.testMask)
                    #Set the merged flag to true
                    merged = True
                    #Break the loop
                    break
            #If the detection has not been merged
            if not merged:
                #Add the detection to the merged detections
                merged_detections.append(detection)
        #Return the merged detections
        return merged_detections

