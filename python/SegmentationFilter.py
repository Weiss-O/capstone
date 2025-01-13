from abc import ABC, abstractmethod
import cv2
import ContourAnalysis as CA
import numpy as np

class SegmentationFilter(ABC):
    @abstractmethod
    def filter(self, image, baseline, proposals) -> list:
        pass

#Basic Segmentation Filter Class Using IOU
class IOUSegmentationFilter(SegmentationFilter):
    def __init__(self, iouThreshold = 0.5):
        self.iouThreshold = iouThreshold

    def filter(self, image, baseline, proposals) -> list:
        label = np.array(1)

        for proposal in proposals:
            masksBaseline, scoresBaseline, logitsBaseline = predictor.predict(
                point_coords=proposal.prompt,
                point_labels=label,
                multimask_output=True #TODO MAY NOT BE NEEDED
            )
        
            masksTest, scoresTest, logitsTest = predictor2.predict(
                point_coords=label,
                point_labels=np.array([1]),
                multimask_output=True
            )

            maskBefore = masksBaseline[np.argmax(scoresBaseline)].astype(bool)
            maskAfter = masksTest[np.argmax(scoresTest)].astype(bool)

            iou = CA.calculate_iou(maskBefore, maskAfter)

            if iou > self.iouThreshold:
                proposals.remove(proposal)

        return proposals