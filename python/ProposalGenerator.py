from abc import ABC, abstractmethod
import cv2
import ProposalPreprocessor as PP
import numpy as np
from skimage.metrics import structural_similarity
import ContourAnalysis as CA
import matplotlib.pyplot as plt

#Constants:


class Proposal():
    def __init__(self, contour, prompt, maskAfter=None):
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
        self.image_diagonal = np.sqrt(self.baseline.shape[0]**2 + self.baseline.shape[1]**2)
        self.baseline_preprocessed = self.preprocess(self.baseline)

    def preprocess(self, image):
        return self.preprocessor.preprocess(image)

    def generateProposals(self, image, **kwargs) -> list:
        kwargs = kwargs
        warp = kwargs.get("warp", False)
        try:
            if warp:
                # Detect the ORB keypoints and descriptors
                orb = cv2.ORB_create()
                kp1, des1 = orb.detectAndCompute(self.baseline, None)
                kp2, des2 = orb.detectAndCompute(image, None)

                # Match keypoints using BFMatcher
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(des1, des2)
                matches = sorted(matches, key=lambda x: x.distance)

                # Ensure there are enough matches
                if len(matches) < 4:
                    raise ValueError("Not enough matches found between the images")

                # Extract point correspondences
                dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                src_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

                # Assert that the number of points matches
                assert len(dst_pts) == len(src_pts), "Number of points does not match"

                # Compute homography (or affine if only rotation/translation)
                M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)

                # Warp image
                warped_image = cv2.warpAffine(image, M, (self.baseline.shape[1], self.baseline.shape[0]))

                mask = (warped_image == 0)
                self.baseline[mask] = 0
                self.baseline_preprocessed = self.preprocess(self.baseline)
                image_preprocessed = self.preprocess(warped_image)
            else:
                image_preprocessed = self.preprocess(image)
        except:
            image_preprocessed = self.preprocess(image)
            warp = False
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

        mask = np.zeros(image_preprocessed.shape, dtype='uint8')
        proposals = []

        bboxes=image_preprocessed.copy()
        bboxes = cv2.cvtColor(bboxes, cv2.COLOR_GRAY2BGR)

        for c in contours:
            #Find contour area
            area = cv2.contourArea(c)
            if area > self.areaThreshold * 0.8:
                rect = cv2.minAreaRect(c)
                (x,y), (w,h), angle = rect
                aspect_ratio = min(w,h)/max(w,h)
                cv2.drawContours(mask, [c], 0, (255,255,255), -1)
                if min(w, h) > 15 and max(w, h) < self.image_diagonal*0.5 and aspect_ratio > 0.15 and w*h > self.areaThreshold and max(w,h) < 0.5*min(self.baseline.shape[1], self.baseline.shape[0]):
                    #Find point on detection area closest to centroid
                    point = CA.get_centroid_safe(c)
                    
                    #Create proposal object
                    proposal = Proposal(c, prompt=point)

                    #Append proposal to list
                    proposals.append(proposal)

                    #Draw rotated rectangle in green
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    cv2.drawContours(bboxes, [box], 0, (0, 255, 0), 10)
                    cv2.circle(bboxes, tuple(point), 50, (255, 0, 0), -1)
                else:
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    cv2.drawContours(bboxes, [box], 0, (0, 255, 0), 10)
        
        print("# unfiltered proposals: ", len(proposals))
        

        if warp:
            # Use matplotlib to show the difference map, the contours, bboxes, and mask
            # fig, ax = plt.subplots(1, 4, figsize=(15, 5))
            # ax[0].imshow(diff, cmap="gray")
            # ax[0].set_title("Difference Map")
            # ax[0].axis("off")
            # ax[1].imshow(threshed_img, cmap="gray")
            # ax[1].set_title("Thresholded Image")
            # ax[1].axis("off")
            # ax[2].imshow(bboxes, cmap="gray")
            # ax[2].set_title("Bounding Boxes")
            # ax[2].axis("off")
            # ax[3].imshow(mask, cmap="gray")
            # ax[3].set_title("Mask")
            # ax[3].axis("off")
            # plt.show()
            return proposals, warped_image
        else:
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
        cv2.circle(image, tuple(proposal.prompt), 5, (0, 0, 255), -1)
    
    cv2.imshow("Proposals", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()