from abc import ABC, abstractmethod

class Object():
    def __init__(self, detection, camera_position):
        self.point = [detection.prompt[0], detection.prompt[1], 1]
        self.camera_position = camera_position 
        self.bbox = None #TODO: Implement bounding box

