from abc import ABC, abstractmethod

class Object():
    def __init__(self, detection, POSID):
        self.ray_position = calculate_ray_position(detection)
        self.POSID = POSID

