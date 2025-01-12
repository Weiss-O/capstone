from abc import ABC, abstractmethod
import cv2

class SegmentationFilter(ABC):
    @abstractmethod
    def filter(self, image, baseline, proposals) -> list:
        pass

#Basic Segmentation Filter Class Using IOU
class IOUSegmentationFilter(SegmentationFilter):
    def __init__(self, iouThreshold = 0.5):
        self.iouThreshold = iouThreshold
    
    def filter(self, image, baseline, proposals) -> list:
        pass