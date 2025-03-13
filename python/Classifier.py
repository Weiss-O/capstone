from abc import ABC, abstractmethod
import os
import cv2
import struct
import numpy as np
import ContourAnalysis as CA
import matplotlib.pyplot as plt

# ----------------------------------
# 1. Create Model Predictor Interface
# ----------------------------------
class PredictorInterface(ABC):
    @abstractmethod
    def set_image(self, image): pass
    
    @abstractmethod
    def predict(self, **kwargs): pass

# ----------------------------------
# 2. Implement SAM2 Predictor Adapter
# ----------------------------------
class SAM2Predictor(PredictorInterface):
    def __init__(self):
        self.predictor = PredictorFactory.create_sam2_predictor(predictor_type="sam2_local")
        
    def set_image(self, image):
        self.predictor.set_image(image)
        
    def predict(self, **kwargs):
        results = []
        for prompt in kwargs.get('prompts', []):
            point = np.array([[prompt[0], prompt[1]]])
            mask, score, _ = self.predictor.predict(
                point_coords=point,
                point_labels=np.array([1]),
                multimask_output=False #TODO MAY NOT BE NEEDED
            ) #The _ is where logits would be returned. We don't need them for now.
            #mask = masks[np.argmax(scores)].astype(bool)
            results.append([mask[0].astype(bool), score[0]])
        
        return results

# ----------------------------------
# 2.5 Implement Remote Predictor Interface
# ----------------------------------
class RemotePredictor(PredictorInterface):
    def __init__(self, socket):
        try:
            self.socket = socket
            command = b'INIT_PREDICTOR'
            self.id = str(id(self)).encode()
            command_length = len(command).to_bytes(4, 'big')
            id_length = len(self.id).to_bytes(4, 'big')

            #Send command to remote server to initialize predictor
            self.socket.sendall(command_length + command)

            if self.get_response() != b'INIT_PREDICTOR_ACK':
                raise Exception("Failed to initialize predictor")
            
            # Send unique predictor ID to server
            self.socket.sendall(id_length + self.id)

            if self.get_response() != b'ID_ACK':
                raise Exception("Failed to register predictor ID")
        except Exception as e:
            print(f"Failed to initialize remote predictor: {e}")

    def set_image(self, image):
        try:
            # Prepare message parts
            command = b'SET_IMAGE'
            command_length = len(command).to_bytes(4, 'big')
            id_length = len(self.id).to_bytes(4, 'big')
            
            # Send everything with lengths
            self.socket.sendall(command_length + command)
            self.socket.sendall(id_length + self.id)
            
            # Encode the image
            success, encoded_image = cv2.imencode('.jpg', image)
            if not success:
                raise Exception("Failed to encode image")
            
            # Convert to bytes
            image_bytes =  encoded_image.tobytes()

            # Send with length header
            size_bytes = len(image_bytes).to_bytes(4, 'big')
            self.socket.sendall(size_bytes + image_bytes)
            
            #Check if image was received
            if self.get_response() != b'SET_IMAGE_ACK':
                raise Exception("Failed to set image")
        except Exception as e:
            print(f"Failed to set image: {e}")

    def predict(self, **kwargs):
        #Make request to remote server
        command = b'PREDICT'
        command_length = len(command).to_bytes(4, 'big')
        id_length = len(self.id).to_bytes(4, 'big')
        self.socket.sendall(command_length + command)
        self.socket.sendall(id_length + self.id)
        # Convert prompts to bytes
        prompts = kwargs.get('prompts', [])
        prompt_bytes = b''
        num_prompts = len(prompts)
        prompt_bytes += num_prompts.to_bytes(4, 'big')
        
        for prompt in prompts:
            x, y = prompt
            prompt_bytes += int(x).to_bytes(4, 'big')
            prompt_bytes += int(y).to_bytes(4, 'big')
            
        # Send prompts
        prompt_len = len(prompt_bytes).to_bytes(4, 'big')
        self.socket.sendall(prompt_len + prompt_bytes)
        
        # Get response #TODO: The response will be the masks, scores, and logits
        response = self.get_response()
        if response != b'PREDICT_ACK':
            raise Exception("Failed to get prediction")
        print("Predict acknowledged")
        results = self.receive_prediction_results()
        return results

    def get_response(self):
        resp_len = int.from_bytes(self.socket.recv(4), 'big') 
        response = self.socket.recv(resp_len)
        return response
    
    def recvall(self, n):
        """Helper function to receive exactly n bytes."""
        data = b''
        while len(data) < n:
            packet = self.socket.recv(n - len(data))
            if not packet:
                #Connection closed unexpectedly
                return None
            data += packet
        return data
    
    def receive_prediction_results(self):
        results = []

        num_results = int.from_bytes(self.recvall(4), 'big')
        print("Expecting results for ", num_results, " prompts")
        for _ in range(num_results):
            height = int.from_bytes(self.recvall(4), 'big')
            width = int.from_bytes(self.recvall(4), 'big')
            print("Expecting mask of size: ", height, "x", width)
            #Calculate expected number of bytes
            mask_size = int.from_bytes(self.recvall(4), 'big')
            print("Expecting mask of size: ", mask_size)
            mask_bytes = self.recvall(mask_size)
            score_bytes = self.recvall(4)
            
            mask=np.frombuffer(mask_bytes, dtype=np.uint8).reshape((height, width)).astype(bool)
            score = struct.unpack('!f', score_bytes)[0]
            results.append([mask, score])
        return results
# ----------------------------------
# 3. Create Predictor Factory
# ----------------------------------
class PredictorFactory:
    @staticmethod
    def create_sam2_predictor(predictor_type, **kwargs):
        if predictor_type == "sam2_local":
            #CUDA Setup
            import torch
            device = torch.device("cuda")
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            
            sam_path = os.path.expanduser("~/sam2")
            
            sam2_checkpoint = os.path.join(sam_path, "checkpoints/sam2.1_hiera_large.pt")
            model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            model = build_sam2(model_cfg, sam2_checkpoint, device) #TODO: Fill in the arguments
            predictor = SAM2ImagePredictor(model)
            return predictor
        elif predictor_type == "sam2_remote":
            return RemotePredictor(socket=kwargs["socket"])

# ----------------------------------
# 4. Segmentation Filter Interface
# ----------------------------------

class Classifier(ABC):
    @abstractmethod
    def classify(self, image, proposals) -> list:
        pass

# ----------------------------------
# 5. Refactored IOU Classifier
# ----------------------------------
class IOUSegmentationClassifier(Classifier):
    def __init__(self,
                 baseline_predictor: PredictorInterface,
                 test_predictor: PredictorInterface,
                 iou_calculator = None,
                 iou_threshold = 0.5,
                 **kwargs):
        self.baseline_predictor = baseline_predictor
        self.test_predictor = test_predictor
        self.iou_calculator = iou_calculator
        if self.iou_calculator is None:
            self.iou_calculator = CA.calculate_iou
        self.iou_threshold = iou_threshold
        self.kwargs = kwargs
        baseline_image = self.kwargs.get("baseline")
        if baseline_image is not None:
            self.baseline_predictor.set_image(baseline_image)     

    def classify(self, image, proposals, **kwargs) -> list:
        self.test_predictor.set_image(image)
        
        baseline_results, test_results = self._process_proposals(proposals)
        
        for i in range(len(proposals)):
            proposals[i].baselineMask = baseline_results[i][0]
            proposals[i].baselineScore = baseline_results[i][1]
            proposals[i].testMask = test_results[i][0]
            proposals[i].testScore = test_results[i][1]

        # classifier_output = image.copy()
        # for proposal in proposals:
        #     int_mask = (proposal.testMask.astype(np.uint8)) * 255
        # #Find largest contour
        #     contours, _ = cv2.findContours(int_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #     #Find bounding rect for largest contour
        #     largest_contour = max(contours, key=cv2.contourArea)
        
        #     # Find bounding rect for the largest contour
        #     x, y, w, h = cv2.boundingRect(largest_contour)
        #     cv2.rectangle(classifier_output, (x, y), (x+w, y+h), (0, 255, 0), 2)
        #     cv2.circle(classifier_output, (proposal.prompt[0], proposal.prompt[1]), 50, (255, 0, 0), -1)
        
        # plt.imshow(classifier_output)
        # plt.show()
        

        classified_proposals = self._apply_iou_filter(proposals)

        return_all = kwargs.get("return_all")
        if return_all is not None:
            classified_proposals = proposals
        
        return classified_proposals
    
    def _process_proposals(self, proposals):
        # Extract prediction logic
        baseline_results = self.baseline_predictor.predict(prompts=[p.prompt for p in proposals])
        test_results = self.test_predictor.predict(prompts = [p.prompt for p in proposals])

        return baseline_results, test_results
        
    
    def _apply_iou_filter(self, processed):
        # Implement threshold logic
        for i in range(len(processed)):
            for j in range(i, len(processed)):
                iou = self.iou_calculator(processed[i].baselineMask, processed[j].testMask)
                if iou > self.iou_threshold:
                    processed[i].iou = 1
                    break
                elif i == j:
                    processed[i].iou = iou

        for i in range(len(processed)):
            if processed[i].iou < self.iou_threshold:
                processed[i].decision = True
            else:
                processed[i].decision = False
        if self.kwargs.get("return_all", False):
            return processed
        else:
            return [p for p in processed if p.decision]

if __name__ == "__main__":
    # Create a remote predictor
    import socket
    import yaml
    
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    HOST = config["server_settings"]["HOST"]
    PORT = config["server_settings"]["PORT"]

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))

    remote_predictor = RemotePredictor(socket=s)
    remote_predictor.set_image("test.jpg")
    remote_predictor.predict(point_coords=[(10, 10), (20, 20)])
    s.close()

    