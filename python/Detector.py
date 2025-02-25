from abc import ABC, abstractmethod
import OPO as OPO
import ProposalGenerator as PG
import Classifier as CL
import NMS as NMS
import cv2
import numpy as np
import yaml
import Server

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

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

        #Convert the masks to a more memory efficient format easy to send over network
        detections = [Detection.from_mask(detection.testMask) for detection in detections]
        return detections
    
    def _mergeDetections(self, detections):
        if self.merger is not None:
            detections = self.merger.merge(detections, iou_threshold=0.5)


import hashlib
import json


def generate_detector_ID(POSID:str, length = 16):
    #Use the POSID to generate a unique hash for the detector
    json_repr = json.dumps(config["baseline"][POSID], sort_keys=True, separators=(',', ':'))
    hash_digest = hashlib.sha256(json_repr.encode()).digest()[:length]
    return hash_digest.hex()

class RemoteDetector(Detector):
    def __init__(self,
                 POSID:str,
                 server):        
        image_path = config["baseline"][POSID]["image_path"]
        self.baseline = cv2.imread(image_path)
        self.server = server
        try:
            #Tell the server we want to init a new detector
            command = b'INIT_DETECTOR'
            self.id = str(generate_detector_ID(POSID=POSID)).encode()
            print(f"Generated ID: {self.id} for position {POSID}")

            Server.send_bytes(self.server, command)
            resp = Server.get_response(self.server)
            if resp != b'INIT_DETECTOR_ACK':
                raise Exception(f"Expected INIT_DETECTOR_ACK but got {resp}")
            
            #Send a unique ID to facilitate request handling
            Server.send_bytes(self.server, self.id)
            resp = Server.get_response(self.server)
            if resp != b'ID_ACK':
                raise Exception(f"Expected ID_ACK but got {resp}")
            
            #Encode and send the baseline image for the detector
            success, encoded_baseline = cv2.imencode('.jpg', self.baseline)
            if not success:
                raise Exception("Error encoding image")
            
            encoded_baseline = encoded_baseline.tobytes()

            Server.send_bytes(self.server, encoded_baseline)
            resp = Server.get_response(self.server)
            if resp != b'BASELINE_ACK':
                raise Exception(f"Expected BASELINE_ACK but got {resp}")

            Server.send_coords(self.server, config["baseline"][POSID]["camera_pos"])

            resp = Server.get_response(self.server)
            if resp != b'POS_ACK':
                raise Exception(f"Expected POS_ACK but got {resp}")

        except Exception as e:
            print(f"Error initializing detector: {e}")
            raise e
        
    def detect(self, imageObj) -> list:
        try:
            command = b'DETECT'
            Server.send_bytes(self.server, command)
            Server.send_bytes(self.server, self.id)
            
            resp = Server.get_response(self.server)
            if resp != b'ID_ACK':
                raise Exception(f"Expected ID_ACK but got {resp}")

            #Encode the image
            success, encoded_image = cv2.imencode('.jpg', imageObj)
            if not success:
                raise Exception("Error encoding image")

            #Convert to bytes
            encoded_image = encoded_image.tobytes()

            #Send image with length header
            Server.send_bytes(self.server, encoded_image)

            detection_data = Server.recv_detections(self.server)
            print(f"Received detections: {detection_data}")
            detections = [Detection.from_array(data) for data in detection_data]

            return detections
        except Exception as e:
            print(f"Error detecting objects: {e}")
            raise e

class Detection():
    def __init__(self, x:int, y:int, w:int, h:int):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
    
    def get_as_array(self):
        """
        Returns an array representation of the detection.
        x, y, are referenced to top left corner of the bounding box.
        
        Returns:
            list: A list containing the x, y, width, and height of the detection.
        """
        return [self.x, self.y, self.w, self.h]
    
    @staticmethod
    def from_mask(mask):
        """
        Creates a Detection object from a binary mask.

        Args:
            mask: A boolean mask (numpy array) where True represents pixels that are part of the object.

        Returns:
            Detection: A new Detection object with the bounding box coordinates that enclose the mask.

        Example:
            >>> mask, score, logit = model.predict(image) #Using SAM2 predictor
            >>> detection = Detection.from_mask(mask)
        """
        int_mask = (mask.astype(np.uint8)) * 255
        x, y, w, h = cv2.boundingRect(int_mask)
        return Detection(x, y, w, h)

    @staticmethod
    def from_array(array):
        return Detection(array[0], array[1], array[2], array[3])

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
