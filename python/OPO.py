from abc import ABC, abstractmethod

class Object():
    def __init__(self, detection, camera_position):
        self.point = detection.prompt
        self.camera_position = camera_position 
        self.bbox = None #TODO: Implement bounding box

