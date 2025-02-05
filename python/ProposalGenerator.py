from abc import ABC, abstractmethod
import cv2
import ProposalPreprocessor as PP
import numpy as np
from skimage.metrics import structural_similarity
import ContourAnalysis as CA

#Constants:


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
    def __init__(self, baseline, preprocessor:PP.ProposalPreprocessor=None, areaThreshold = 0):
        self.preprocessor = preprocessor if preprocessor != None else PP.GrayscaleGaussianPreprocessor()
        self.baseline = baseline
        self.areaThreshold = areaThreshold

        self.baseline_preprocessed = self.preprocess(self.baseline)

    def preprocess(self, image):
        return self.preprocessor.preprocess(image)

    def generateProposals(self, image) -> list:
        image_preprocessed = self.preprocess(image)

        #Compute the SSIM between the two images
        score, diff = structural_similarity(
            self.baseline_preprocessed, image_preprocessed, full=True
        )

        print("SSIM: ", score)

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

        for c in contours:
            if cv2.contourArea(c) > self.areaThreshold:
                #Find point on detection area closest to centroid
                point = np.array([CA.get_centroid_safe(c)])
                
                #Create proposal object
                proposal = Proposal(c, prompt=point)

                #Append proposal to list
                proposals.append(proposal)
        
        print("# unfiltered proposals: ", len(proposals))
        return proposals

if __name__ == "__main__":
    import os
    root_dir = os.path.dirname(os.path.abspath(__file__))+ "/"



    pGen = SSIMProposalGenerator(
        baseline = cv2.imread(os.path.join(root_dir, "test_set/capture_2.jpg")),
        preprocessor=PP.GrayscaleGaussianPreprocessor(),
        areaThreshold=20
    )

    image = cv2.imread(os.path.join(root_dir, "test_set/capture_10.jpg"))
    processed_image = pGen.preprocess(image)
    
    
    # #Show the processed image
    # processed_image_normalized = cv2.normalize(processed_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # cv2.imshow("Processed Image", processed_image_normalized)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    proposals = pGen.generateProposals(cv2.imread(os.path.join(root_dir, "test_set/capture_10.jpg")))
    #For each proposal, add a bounding box to the image
    for proposal in proposals:
        cv2.drawContours(image, [proposal.contour], -1, (0, 255, 0), 2)
        cv2.circle(image, tuple(proposal.prompt[0]), 5, (0, 0, 255), -1)
    
    cv2.imshow("Proposals", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()