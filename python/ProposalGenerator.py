from abc import ABC, abstractmethod
import cv2
import ProposalPreprocessor
import numpy as np
from skimage.metrics import structural_similarity
import ContourAnalysis as CA

class Proposal():
    def __init__(self, contour, prompt):
        self.contour = contour
        self.prompt = prompt

class ProposalGenerator(ABC):
    @abstractmethod
    def generateProposals(self, image) -> list:
        pass

#Basic Proposal Generator Class Using SSIM
class SSIMProposalGenerator(ProposalGenerator):
    def __init__(self, preprocessor:ProposalPreprocessor.ProposalPreprocessor=None, baseline=None):
        self.preprocessor = preprocessor if preprocessor != None else ProposalPreprocessor.NullPreprocessor()
        self.baseline = baseline if baseline != None else cv2.imread("baseline.jpg")

        self.baseline_preprocessed = self.preprocess(self.baseline)

    def preprocess(self, image):
        return self.preprocessor.preprocess(image)

    def generateProposals(self, image) -> list:
        image_preprocessed = self.preprocess(image)

        #Compute the SSIM between the two images
        _, diff = structural_similarity(
            self.baseline_preprocessed, image_preprocessed, full=True
        )

        diff = (diff*255).astype("uint8")
        thresh, threshed_img = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        #Check how close diff values are to threshold (is otsu appropriate)
        mean_diff_value = np.mean(np.abs(diff-thresh))

        #Determine if Otsu thresholding is appropriate
        if mean_diff_value < 10:
            threshed_img = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY_INV)[1]

        #Contour Analysis
        contours, _ = cv2.findContours(threshed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        proposals = []

        for contour in contours:
            point = np.array([CA.get_centroid_safe(contour)])
            proposal = Proposal(contour, point)
            proposals.append(proposal)
        
        return proposals

if __name__ == "__main__":
    pGen = SSIMProposalGenerator(
        ProposalPreprocessor.GrayscaleGaussianPreprocessor()
    )

    processed_image = pGen.preprocess(cv2.imread("test_images.jpg"))