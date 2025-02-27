import cv2
import numpy as np

#Load the images
img1 = cv2.imread('baseline1.jpg',0)
img2 = cv2.imread('test.jpg',0)

#Detect the ORB keypoints and descriptors
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# Match keypoints using FLANN
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# Extract point correspondences
src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# Compute homography (or affine if only rotation/translation)
M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)

# Warp image
aligned = cv2.warpAffine(img2, M, (img1.shape[1], img1.shape[0]))

cv2.imwrite("aligned.jpg", aligned)