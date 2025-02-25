# Description: This file is the main file for the detector server.
# It handles client requests for creating detectors and all object detection

import cv2
import numpy as np
import socket
import yaml

import Server
import Detector

with open('config.yaml') as file:
    config = yaml.safe_load(file)

import Classifier
import PersonDetection
import ProposalGenerator as PG
# import OPO

detectors={}
# detectedObjects = []

testPredictor = Classifier.SAM2Predictor()

def create_detector(baseline_image):
    baselinePredictor = Classifier.SAM2Predictor()

    classifier = Classifier.IOUSegmentationClassifier(baseline_predictor=baselinePredictor,
                                                        test_predictor=testPredictor,
                                                        iou_threshold = 0.5,
                                                        baseline=baseline_image)

    proposal_generator = PG.SSIMProposalGenerator(baseline=baseline_image,
                                                    areaThreshold=1000)
    detector = Detector.BasicDetector(baseline=baseline_image,
                                        proposal_generator=proposal_generator,
                                        classifier=classifier)
    return detector

#Function to detect objects in an image from prompts
def handle_client(client_socket):
    try:
        while True:
            command = Server.get_response(client_socket)
            if command is None:
                print("Connection closed unexpectedly")
                break
            print(f"Received command: {command}")

            if command == b'INIT_DETECTOR':
                #Acknowledge request
                Server.send_bytes(client_socket, b'INIT_DETECTOR_ACK')

                #Receive the detector ID
                detector_id = Server.get_response(client_socket)
                print(f"Initializing new detector with ID: {detector_id}")

                #Acknowledge the ID
                Server.send_bytes(client_socket, b'ID_ACK')

                #Receive the baseline image
                baseline_bytes = Server.get_response(client_socket)
                baseline = cv2.imdecode(np.frombuffer(baseline_bytes, np.uint8), cv2.IMREAD_COLOR)
                print(f"Received baseline image of size {len(baseline_bytes)} bytes")

                if not(detector_id in detectors.keys()):
                    #Create the detector
                    detectors[detector_id] = create_detector(baseline)

                #Acknowledge creation of the detector
                Server.send_bytes(client_socket, b'BASELINE_ACK')

            elif command == b'DETECT':
                #Receive the detector ID
                detector_id = Server.get_response(client_socket)
                print(f"Received detection request from detector {detector_id}")

                #Acknowledge the ID
                Server.send_bytes(client_socket, b'ID_ACK')

                #Receive the image
                image_bytes = Server.get_response(client_socket)
                image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
                print(f"Received image of size {len(image_bytes)} bytes")

                #Detect objects in the image
                detections = detectors[detector_id].detect(image)

                #TODO: Send the detections

                #Send back the detections
                detections_json = json.dumps(detections)
                detections_bytes = detections_json.encode()
                Server.send_bytes(client_socket, detections_bytes)

            else:
                print(f"Unknown command: {command}")
                break
    except Exception as e:
        print(f"Error handling client: {e}")
    finally:
        client_socket.close()

if __name__ == "__main__":
    host = config['server_settings']['HOST']
    port = config['server_settings']['PORT']

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen()

    print(f"Server listening on {host}:{port}")

    while True:
        client_socket, addr = server_socket.accept()
        print(f"Connection from {addr}")
        handle_client(client_socket)