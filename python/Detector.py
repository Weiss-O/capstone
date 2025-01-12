from abc import ABC, abstractmethod
import python.OPO as OPO
import ProposalGenerator as PG
import SegmentationFilter as SF

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
    def __init__(self, baseline, proposalGenerator=None, segmentationFilter=None):
        self.baseline = baseline
        if proposalGenerator is None:
            self.proposalGenerator = PG.SSIMProposalGenerator()
        if segmentationFilter is None:
            self.segmentationFilter = SF.IOUSegmentationFilter()

    #Function to take in image and generate list of objects
    def detect(self, imageObj) -> list:
        proposals = self.proposalGenerator.generateProposals(imageObj, self.baseline)
        return self.segmentationfilter.filter(imageObj, self.baseline, proposals)
    
if __name__ == "__main__":
    baseline = r"test_set/capture_2.jpg:"
    image = r"test_set/capture_10.jpg"

    detector = BasicDetector(baseline)