import numpy as np
from abc import ABC, abstractmethod

class Coordinate(ABC):
    @abstractmethod
    def __init__(self):
        pass

class pixel_coordinate(Coordinate):
    def __init__(self, x, y):
        self.x = x
        self.y = y

class pointing_coordinate(Coordinate):
    def __init__(self, theta, phi):
        self.theta = theta
        self.phi = phi

