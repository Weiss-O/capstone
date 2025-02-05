from abc import ABC, abstractmethod
import cv2
import ContourAnalysis as CA
import numpy as np
import torch
import os

class SegmentationFilter(ABC):
    @abstractmethod
    def filter(self, image, proposals) -> list:
        pass

#Basic Segmentation Filter Class Using IOU
class IOUSegmentationFilter(SegmentationFilter):
    def __init__(self, baseline, iouThreshold = 0.5):
        self.iouThreshold = iouThreshold
        

        device = torch.device("cuda")
        #use bfloat16
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # Turn on tfloat32 for Ampere GPUs (Lenovo Legion has one)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor


        sam_path = os.path.expanduser("~/sam2")
        
        sam2_checkpoint = os.path.join(sam_path, "checkpoints/sam2.1_hiera_large.pt")
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

        self.sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
        self.baseline_predictor = SAM2ImagePredictor(self.sam2_model)
        self.test_predictor = SAM2ImagePredictor(self.sam2_model)

        self.baseline_predictor.set_image(baseline)

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
        label = np.array([1])
        filtered_proposals = []

        self.test_predictor.set_image(image)
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


# -------------------------------------
# Class to substitute for segmentation filter when processing is done server-side
# -------------------------------------
class remoteSegmentationFilter(SegmentationFilter): #TODO: This class will not work in its current state under the Liskov Subquestion Principle
    def __init__(self, server_adapter):
        self.server = server_adapter

    def filter(self, image, proposals) -> list:
        
        return self.server.process(image, proposals)
