from abc import ABC, abstractmethod
import OPO as OPO
import ProposalGenerator as PG
import SegmentationFilter as SF
import cv2

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
                baseline=self.baseline
            )
        if self.segmentation_filter is None:
            self.segmentation_filter = SF.IOUSegmentationFilter()

    #Function to take in image and generate list of objects
    def detect(self, imageObj) -> list:
        proposals = self.proposal_generator.generateProposals(imageObj)
        return self.segmentation_filter.filter(imageObj, self.baseline, proposals)
    
if __name__ == "__main__":
    baseline_path = r"test_set/capture_2.jpg"
    image_path = r"test_set/capture_10.jpg"

    detector = BasicDetector(baseline=cv2.imread(baseline_path))
    detections = detector.detect(cv2.imread(image_path))