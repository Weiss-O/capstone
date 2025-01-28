from abc import ABC, abstractmethod
import cv2
import ContourAnalysis as CA
import numpy as np
import torch
import os

# ----------------------------------
# 1. Create Model Predictor Interface
# ----------------------------------
class Predictor(ABC):
    @abstractmethod
    def set_image(self, image): pass
    
    @abstractmethod
    def predict(self, point_coords, point_labels, multimask_output): pass

# ----------------------------------
# 2. Implement SAM2 Predictor Adapter
# ----------------------------------
class SAM2Predictor(Predictor):
    def __init__(self, model):
        self.predictor = PredictorFactory.create_sam2_predictor(model)
        
    def set_image(self, image):
        self.predictor.set_image(image)
        
    def predict(self, **kwargs):
        return self.predictor.predict(**kwargs)
    
# ----------------------------------
# 3. Create Predictor Factory
# ----------------------------------
class PredictorFactory:
    @staticmethod
    def create_sam2_predictor(baseline_image):
        device = torch.device("cuda")
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # sam_path = os.path.expanduser("~/sam2")
        
        # sam2_checkpoint = os.path.join(sam_path, "checkpoints/sam2.1_hiera_large.pt")
        # model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        model = build_sam2(...) #TODO: Fill in the arguments
        predictor = SAM2Predictor(model)
        predictor.set_image(baseline_image)
        return predictor

# ----------------------------------
# 4. Segmentation Filter Interface
# ----------------------------------

class SegmentationFilter(ABC):
    @abstractmethod
    def filter(self, image, proposals) -> list:
        pass

# ----------------------------------
# 5. Refactored IOU Filter with DI
# ----------------------------------
class IOUSegmentationFilter(SegmentationFilter):
    def __init__(self,
                 baseline_predictor: Predictor,
                 test_predictor: Predictor,
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
    
class remoteSegmentationFilter(SegmentationFilter):
    def __init__(self, server, POSID):
        self.server = server
        self.POSID = POSID

    def filter(self, image, proposals) -> list:
        pass