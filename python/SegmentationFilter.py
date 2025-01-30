from abc import ABC, abstractmethod
import cv2
import ContourAnalysis as CA
import numpy as np
import torch
import os

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
        return self.predictor.predict(**kwargs)

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

    def set_image(self, image_path):
        try:
            # Prepare message parts
            command = b'SET_IMAGE'
            command_length = len(command).to_bytes(4, 'big')
            id_length = len(self.id).to_bytes(4, 'big')
            
            # Send everything with lengths
            self.socket.sendall(command_length + command)
            self.socket.sendall(id_length + self.id)
            
            # Check size of image
            image_size = os.path.getsize(image_path)
            size_bytes = image_size.to_bytes(4, 'big')
            self.socket.sendall(size_bytes)

            #Stream the file in chunks
            chunk_size = 8192
            with open(image_path, 'rb') as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    self.socket.sendall(chunk)
            
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
        prompts = kwargs.get('point_coords', [])
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

    def get_response(self):
        resp_len = int.from_bytes(self.socket.recv(4), 'big') 
        response = self.socket.recv(resp_len)
        return response
# ----------------------------------
# 3. Create Predictor Factory
# ----------------------------------
class PredictorFactory:
    @staticmethod
    def create_sam2_predictor(predictor_type, **kwargs):
        if predictor_type == "sam2_local":
            #CUDA Setup
            device = torch.device("cuda")
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            
            sam_path = os.path.expanduser("~/sam2")
            
            sam2_checkpoint = os.path.join(sam_path, "checkpoints/sam2.1_hiera_large.pt")
            model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            model = build_sam2(model_cfg, sam2_checkpoint, device) #TODO: Fill in the arguments
            predictor = SAM2Predictor(model)
            return predictor
        elif predictor_type == "sam2_remote":
            return RemotePredictor(socket=kwargs["socket"])

# ----------------------------------
# 4. Segmentation Filter Interface
# ----------------------------------

class SegmentationFilter(ABC):
    @abstractmethod
    def filter(self, image, proposals) -> list:
        pass

"""
# ----------------------------------
# 5. Refactored IOU Filter with DI
# ----------------------------------
class IOUSegmentationFilter(SegmentationFilter):
    def __init__(self,
                 baseline_predictor: PredictorInterface,
                 test_predictor: PredictorInterface,
                 iou_calculator,
                 merger,
                 iou_threshold = 0.5):
        self.baseline_predictor = baseline_predictor
        self.test_predictor = test_predictor
        self.iou_calculator = iou_calculator
        self.merger = merger
        self.iou_threshold = iou_threshold

    def merge_proposals(self, proposals):
        #Loop through proposals and merge if IOU for mnaskAfter is above threshold. Repeat until no more merges occur.
        merged = True
        while merged:
            merged = False
            for i in range(len(proposals)):
                for j in range(i+1, len(proposals)):
                    iou = CA.calculate_iou(proposals[i].maskAfter, proposals[j].maskAfter)
                    if iou > 0.8:
                        #Merge the two proposals
                        proposals[i].maskAfter = np.logical_or(proposals[i].maskAfter, proposals[j].maskAfter)
                        proposals[i].prompt = np.round(np.mean([proposals[i].prompt, proposals[j].prompt], axis=0)).astype(int)
                        del proposals[j]
                        merged = True
                        break
                if merged:
                    break

        

    def filter(self, image, proposals) -> list:
        self.test_predictor.set_image(image)
        
        processed = [self._process_proposal(p) 
                    for p in proposals]
        
        filtered = self._apply_iou_filter(processed)
        merged = self.merger.merge(filtered)

        return merged
        
        for proposal in proposals:
            masksBaseline, scoresBaseline, logitsBaseline = self.baseline_predictor.predict(
                point_coords=proposal.prompt,
                point_labels=label,
                multimask_output=True #TODO MAY NOT BE NEEDED
            )
        
            masksTest, scoresTest, logitsTest = self.test_predictor.predict(
                point_coords=proposal.prompt,
                point_labels=np.array([1]),
                multimask_output=True
            )

            maskBefore = masksBaseline[np.argmax(scoresBaseline)].astype(bool)
            maskAfter = masksTest[np.argmax(scoresTest)].astype(bool)

            iou = CA.calculate_iou(maskBefore, maskAfter)

            proposal.maskAfter = maskAfter
            proposal.maskBefore = maskBefore
            proposal.iou = iou
            filtered_proposals.append(proposal)
        
        
        #loop through. if the mask AFTER for a proposal has high IOU (0.8) with the mask BEFORE of any other proposal, set its iou = 1
        for i in range(len(filtered_proposals)):
            for j in range(i+1, len(filtered_proposals)):
                iou = CA.calculate_iou(filtered_proposals[i].maskAfter, filtered_proposals[j].maskBefore)
                if iou > self.iouThreshold:
                    filtered_proposals[i].iou = 1
                    break
        
        #filter out iou greater than threshold
        filtered_proposals = [proposal for proposal in filtered_proposals if proposal.iou < self.iouThreshold]

        #merge proposals which share maskAfter
        self.merge_proposals(filtered_proposals)

        print("# Filtered Proposals: ", len(filtered_proposals))
        return filtered_proposals
    
    def _process_proposal(self, proposal):
        # Extract prediction logic
        baseline_masks = self.baseline_predictor.predict(...) #TODO: Fill in the arguments
        test_masks = self.test_predictor.predict(...) #TODO: Fill in the arguments
        
        return ProcessedProposal(
            mask_before=baseline_masks[np.argmax(scores)],
            mask_after=test_masks[np.argmax(scores)],
            proposal=proposal
        )
    
    def _apply_iou_filter(self, processed):
        # Implement threshold logic
        return [p for p in processed 
               if self.iou_calculator(p) < self.iou_threshold]
"""

class remoteSegmentationFilter(SegmentationFilter):
    def __init__(self, server, POSID):
        self.server = server
        self.POSID = POSID

    def filter(self, image, proposals) -> list:
        pass

if __name__ == "__main__":
    # Create a remote predictor
    import socket
    import yaml
    
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    HOST = config["server"]["host"]
    PORT = config["server"]["port"]
    