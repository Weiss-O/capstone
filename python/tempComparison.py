from abc import ABC, abstractmethod
import cv2
import ContourAnalysis as CA  # Could be abstracted further
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
        self.predictor = SAM2ImagePredictor(model)
        
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
        # [Move all torch setup here...]
        
        from sam2.build_sam import build_sam2
        model = build_sam2(...)
        predictor = SAM2Predictor(model)
        predictor.set_image(baseline_image)
        return predictor

# ----------------------------------
# 4. Improve SegmentationFilter Hierarchy
# ----------------------------------
class SegmentationFilter(ABC):
    @abstractmethod
    def filter(self, image, proposals) -> list: pass

# ----------------------------------
# 5. Refactored IOU Filter with DI
# ----------------------------------
class IOUSegmentationFilter(SegmentationFilter):
    def __init__(self, 
                 baseline_predictor: Predictor,
                 test_predictor: Predictor,
                 iou_calculator,
                 merger,
                 iou_threshold=0.5):
        self.baseline_predictor = baseline_predictor
        self.test_predictor = test_predictor
        self.iou_calculator = iou_calculator
        self.merger = merger
        self.iou_threshold = iou_threshold

    def filter(self, image, proposals) -> list:
        self.test_predictor.set_image(image)
        
        processed = [self._process_proposal(p) 
                    for p in proposals]
        
        filtered = self._apply_iou_filter(processed)
        merged = self.merger.merge(filtered)
        
        return merged

    def _process_proposal(self, proposal):
        # Extract prediction logic
        baseline_masks = self.baseline_predictor.predict(...)
        test_masks = self.test_predictor.predict(...)
        
        return ProcessedProposal(
            mask_before=baseline_masks[np.argmax(scores)],
            mask_after=test_masks[np.argmax(scores)],
            proposal=proposal
        )

    def _apply_iou_filter(self, processed):
        # Implement threshold logic
        return [p for p in processed 
               if self.iou_calculator(p) < self.threshold]

# ----------------------------------
# 6. Strategy Pattern for Merging
# ----------------------------------
class MergeStrategy(ABC):
    @abstractmethod
    def merge(self, proposals): pass

class IouMergeStrategy(MergeStrategy):
    def __init__(self, threshold=0.8):
        self.threshold = threshold
        
    def merge(self, proposals):
        # Implement merging logic
        return merged_proposals

# ----------------------------------
# 7. Implement Remote Filter Properly
# ----------------------------------
class RemoteSegmentationFilter(SegmentationFilter):
    def __init__(self, server_adapter):
        self.server = server_adapter
        
    def filter(self, image, proposals):
        # Actual implementation using server
        return self.server.process(image, proposals)

# ----------------------------------
# 8. Configuration Management
# ----------------------------------
class FilterConfig:
    def __init__(self, iou_threshold=0.5, merge_threshold=0.8):
        self.iou_threshold = iou_threshold
        self.merge_threshold = merge_threshold