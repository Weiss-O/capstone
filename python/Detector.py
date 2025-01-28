from abc import ABC, abstractmethod
import OPO as OPO
import ProposalGenerator as PG
import SegmentationFilter as SF
import cv2
import numpy as np

class Detector(ABC):
    @abstractmethod
    def detect(self, imageObj) -> list:
        pass

#Basic Detector Class for before after comparison
#   Baseline image is passed to detector during init
#       One baseline image per detector instance
#   Image and baseline are passed to proposalGen and segFilter
#       Unsure whether proposalGen and SegFilter should be classes or class instances
class BasicDetector(Detector):
    def __init__(self, baseline, proposal_generator:PG.ProposalGenerator=None, segmentation_filter:SF.SegmentationFilter=None):
        self.baseline = baseline
        self.segmentation_filter = segmentation_filter
        self.proposal_generator = proposal_generator

        if self.proposal_generator is None:
            self.proposal_generator = PG.SSIMProposalGenerator(
                baseline=self.baseline, areaThreshold=100
            )
        if self.segmentation_filter is None:
            self.segmentation_filter = SF.IOUSegmentationFilter(baseline=self.baseline, iouThreshold=0.7)

    #Function to take in image and generate list of objects
    def detect(self, imageObj) -> list:
        proposals = self.proposal_generator.generateProposals(imageObj)
        proposals = self.segmentation_filter.filter(imageObj, proposals)
        return proposals

    
if __name__ == "__main__":
    import os
    root_dir = os.path.dirname(os.path.abspath(__file__))+ "/"

    baseline = cv2.imread(os.path.join(root_dir, "test_set/capture_2.jpg"))
    image = cv2.imread(os.path.join(root_dir, "test_set/capture_17.jpg"))



    detector = BasicDetector(baseline=baseline)
    detections = detector.detect(imageObj = image)
    #Create a copy of the baseline image for showing maskBefore
    baseline_vis = baseline.copy()

    #Draw the bounding boxes and prompt points for the detections
    for detection in detections:
        #Generate random color once per detection
        random_color = tuple(map(int, np.random.randint(0, 255, size=3)))
        
        #Draw maskAfter on current image
        overlay = image.copy()
        overlay[detection.maskAfter] = random_color
        cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)

        #Draw maskBefore on baseline image
        baseline_overlay = baseline_vis.copy()
        baseline_overlay[detection.maskBefore] = random_color
        cv2.addWeighted(baseline_overlay, 0.5, baseline_vis, 0.5, 0, baseline_vis)

        #Draw prompt point and contour with same color as mask
        cv2.circle(image, tuple(detection.prompt[0]), 5, random_color, -1)
        cv2.drawContours(image, [detection.contour], -1, random_color, 3)

    cv2.imshow("Detections", image)
    cv2.imshow("Original Objects", baseline_vis)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
