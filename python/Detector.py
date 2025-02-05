from abc import ABC, abstractmethod
import OPO as OPO
import ProposalGenerator as PG
import Classifier as CL
import NMS as NMS
import cv2
import numpy as np

# import config

class Detector(ABC):
    @abstractmethod
    def detect(self, imageObj) -> list:
        pass

#Basic Detector Class for before after comparison
#   Baseline image is passed to detector during init
#       One baseline image per detector instance
#   Image and baseline are passed to proposalGen and Classifier
#       Unsure whether proposalGen and Classifier should be classes or class instances
class BasicDetector(Detector):
    def __init__(self,
                 baseline,
                 proposal_generator:PG.ProposalGenerator,
                 classifier:CL.Classifier,
                 merger:NMS.Merger = None):
        self.baseline = baseline
        self.classifier = classifier
        self.proposal_generator = proposal_generator
        self.merger = merger

    #Function to take in image and generate list of objects
    def detect(self, imageObj) -> list:
        proposals = self.proposal_generator.generateProposals(imageObj)
        detections = self.classifier.classify(imageObj, proposals)
        self._mergeDetections(detections)
        return detections
    
    def _mergeDetections(self, detections):
        if self.merger is not None:
            detections = self.merger.merge(detections, iou_threshold=0.5)

    
if __name__ == "__main__":
    import os
    # import socket
    # import yaml

    # with open("config.yaml", "r") as file:
    #     config = yaml.safe_load(file)

    # HOST = config["server_settings"]["HOST"]
    # PORT = config["server_settings"]["PORT"]
    # s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # s.connect((HOST, PORT))
    
    root_dir = os.path.dirname(os.path.abspath(__file__))+ "/"

    baseline = cv2.imread(os.path.join(root_dir, "test_images/fr_baseline.jpg"))
    image = cv2.imread(os.path.join(root_dir, "test_images/fr_test.jpg"))


    classifier = CL.IOUSegmentationClassifier(
         baseline_predictor=CL.SAM2Predictor(),
         test_predictor=CL.SAM2Predictor(),
         baseline=baseline)         

    detector = BasicDetector(baseline=baseline,
                             proposal_generator=PG.SSIMProposalGenerator(baseline = baseline,
                                                                          areaThreshold=400),
                             classifier=classifier,
                             merger=NMS.TestMaskIOUMerger)
    
    detections = detector.detect(imageObj=image)

    print ([detection.prompt for detection in detections])

    """
    #Create a copy of the baseline image for showing maskBefore
    baseline_vis = baseline.copy()

    #Draw the bounding boxes and prompt points for the detections
    for detection in detections:
        #Generate random color once per detection
        random_color = tuple(map(int, np.random.randint(0, 255, size=3)))
        
        #Draw maskAfter on current image
        overlay = image.copy()
        overlay[detection.testMask] = random_color
        cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)

        #Draw maskBefore on baseline image
        baseline_overlay = baseline_vis.copy()
        baseline_overlay[detection.baselineMask] = random_color
        cv2.addWeighted(baseline_overlay, 0.5, baseline_vis, 0.5, 0, baseline_vis)

        #Draw prompt point and contour with same color as mask
        cv2.circle(image, tuple(detection.prompt), 5, random_color, -1)
        cv2.drawContours(image, [detection.contour], -1, random_color, 3)

    cv2.imshow("Detections", image)
    cv2.imshow("Original Objects", baseline_vis)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
    """
