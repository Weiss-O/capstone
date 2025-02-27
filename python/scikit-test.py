from skimage.metrics import structural_similarity
import cv2
import numpy as np
from skimage import feature
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def dilation(threshed_img, kernel_size=3):
    kernel = np.ones((kernel_size,kernel_size),np.uint8)
    dilation = cv2.dilate(threshed_img,kernel,iterations = 1)
    return dilation

def erosion(threshed_img, kernel_size=3):
    kernel = np.ones((kernel_size,kernel_size),np.uint8)
    erosion = cv2.erode(threshed_img,kernel,iterations = 1)
    return erosion

def fill(threshed_img):
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(threshed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Create a mask to fill the gaps
    mask = np.zeros(threshed_img.shape, dtype=np.uint8)

    # Iterate through contours to fill gaps
    for contour in contours:
        # Draw the contour on the mask
        cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)

    # Perform morphological closing to fill gaps
    kernel = np.ones((5, 5), np.uint8)
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Combine the closed mask with the original thresholded image
    filled_img = cv2.bitwise_or(threshed_img, closed_mask)

    # Fill the contours
    filled_img = cv2.drawContours(threshed_img.copy(), contours, -1, (255), thickness=cv2.FILLED)

    return filled_img

# Function to convert image to local binary pattern map
def lbp_map(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply LBP operation
    lbp = feature.local_binary_pattern(gray, 1, 1, method="uniform")
    return lbp

# Load images
before = cv2.imread('output/alignment_test_images/NP0.00T18.40_OP0T0.jpg')
after = cv2.imread('output/alignment_test_images/NP0.00T18.40_OP0T5.jpg')

#Detect the ORB keypoints and descriptors
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(before,None)
kp2, des2 = orb.detectAndCompute(after,None)

# Match keypoints using FLANN
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# Extract point correspondences
dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
src_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# Compute homography (or affine if only rotation/translation)
M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)

# Warp image
after = cv2.warpAffine(after, M, (before.shape[1], before.shape[0]))

# Convert images to grayscale
before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise and improve differencing
before_gray = cv2.GaussianBlur(before_gray, (5, 5), 0)
after_gray = cv2.GaussianBlur(after_gray, (5, 5), 0)

# Compute SSIM between the two images
(score, diff) = structural_similarity(before_gray, after_gray, full=True)
print("Image Similarity: {:.4f}%".format(score * 100))

# The diff image contains the actual image differences between the two images
# and is represented as a floating point data type in the range [0,1] 
# so we must convert the array to 8-bit unsigned integers in the range
# [0,255] before we can use it with OpenCV
diff = (diff * 255).astype("uint8")
diff_box = cv2.merge([diff, diff, diff])

thresh, threshed_img = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

# Check how close the diff values are to the threshold on average
mean_diff_value = np.mean(np.abs(diff-thresh))
print("Mean diff value: {:.2f}".format(mean_diff_value))
print("Otsu's threshold value: {:.2f}".format(thresh))

# Determine if Otsu's thresholding is appropriate
if mean_diff_value > 10:  # You can adjust this threshold value as needed
    print("Otsu's thresholding is appropriate.")
    #Display a plot of the otsu distribution
    plt.hist(diff.ravel(),256,[0,256]); plt.show()
else:
    threshed_img = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV)[1]

contours = cv2.findContours(threshed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]

# Copy of thresholded image to test processing
threshed_img_dilated = threshed_img.copy()

#reject small changes
for c in contours:
    if cv2.contourArea(c) < 40:
        cv2.drawContours(threshed_img_dilated, [c], -1, (0,0,0), -1) #Remove small contours

# # Dilation to fill small gaps
# threshed_img_dilated = erosion(threshed_img=threshed_img) #break up thin lines and edges (hopefully)

# contours = cv2.findContours(threshed_img_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# contours = contours[0] if len(contours) == 2 else contours[1]

# #reject small changes
# for c in contours:
#     if cv2.contourArea(c) < 100:
#         cv2.drawContours(threshed_img_dilated, [c], -1, (0,0,0), -1) 

threshed_img_dilated = dilation(threshed_img=threshed_img_dilated, kernel_size=10)

# Fill contours

#erosion to remove noise

contours = cv2.findContours(threshed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]

contours_dil = cv2.findContours(threshed_img_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_dil = contours_dil[0] if len(contours_dil) == 2 else contours_dil[1]

mask = np.zeros(before.shape, dtype='uint8')
filled_after = after.copy()

mask_dil = np.zeros(before.shape, dtype='uint8')
# List to store areas of contours
contour_areas = []

lbp_img = lbp_map(before)
lbp_img = cv2.merge([lbp_img, lbp_img, lbp_img])

#make copies of before and after image to display rotated bounding boxes
before_rotated = before.copy()
after_rotated = after.copy()

image_diagonal = np.sqrt(before.shape[0]**2 + before.shape[1]**2)
min_area = 10000

def check_rotated_rect(width, height, aspect_ratio):
    if min(width, height) > 10 and max(width, height) < image_diagonal*0.8 and aspect_ratio > 0.2 and width*height > min_area:
        return True

for c in contours:
    area = cv2.contourArea(c)
    if area > min_area:
        contour_areas.append(area)
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(before, (x, y), (x + w, y + h), (36,255,12), 2)
        cv2.rectangle(after, (x, y), (x + w, y + h), (36,255,12), 2)
        cv2.rectangle(diff_box, (x, y), (x + w, y + h), (36,255,12), 2)
        cv2.drawContours(mask, [c], 0, (255,255,255), -1)
        #print in small font the contour area next to the contour
        cv2.putText(mask, str(area), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        cv2.drawContours(filled_after, [c], 0, (0,255,0), -1)
        # Draw the contour on the lbp image with transparency
        overlay = lbp_img.copy()
        cv2.drawContours(overlay, [c], 0, (0,0,255), -1)
        alpha = 0.3  # Transparency factor
        cv2.addWeighted(overlay, alpha, lbp_img, 1 - alpha, 0, lbp_img)

        rect = cv2.minAreaRect(c)
        (x,y), (w,h), angle = rect
        aspect_ratio = min(w,h)/max(w,h)
        
        color = (0,0,255)
        if check_rotated_rect(w, h, aspect_ratio):
            #set color to green
            color = (0,255,0)
        
        #Draw the rotated rectangle on each image
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        cv2.drawContours(before_rotated,[box],0,color,2)
        cv2.drawContours(after_rotated,[box],0,color,2)




for c in contours_dil:
    area = cv2.contourArea(c)
    if area > 0:
        cv2.drawContours(mask_dil, [c], 0, (255,255,255), -1)

# # Resize images for display
# before_resized = cv2.resize(before, (600, 400))
# after_resized = cv2.resize(after, (600, 400))
# diff_resized = cv2.resize(diff, (600, 400))
# diff_box_resized = cv2.resize(diff_box, (600, 400))
# mask_resized = cv2.resize(mask, (600, 400))
# filled_after_resized = cv2.resize(filled_after, (600, 400))


# Display images
# Create figure with 2x4 subplots
fig, axs = plt.subplots(2, 5, figsize=(20, 10))
fig.suptitle('Image Processing Steps - (0, 5) With Affine Transformation for Alignment')

# Convert BGR to RGB for matplotlib display
before_rgb = cv2.cvtColor(before, cv2.COLOR_BGR2RGB)
after_rgb = cv2.cvtColor(after, cv2.COLOR_BGR2RGB)
filled_after_rgb = cv2.cvtColor(filled_after, cv2.COLOR_BGR2RGB)

# Plot images
axs[0,0].imshow(before_rgb)
axs[0,0].set_title('Before')

axs[0,1].imshow(after_rgb)
axs[0,1].set_title('After')

axs[0,2].imshow(diff, cmap='gray')
axs[0,2].set_title('Difference')

axs[0,3].imshow(diff_box)
axs[0,3].set_title('Difference with Boxes')

axs[0,4].imshow(cv2.cvtColor(before_rotated, cv2.COLOR_BGR2RGB))
axs[0,4].set_title('Before Rotated Rect')

axs[1,0].imshow(mask)
axs[1,0].set_title('Mask')

axs[1,1].imshow(mask_dil)
axs[1,1].set_title('Dilated Mask')

axs[1,2].imshow(filled_after_rgb)
axs[1,2].set_title('Filled After')

axs[1,3].imshow(lbp_img)
axs[1,3].set_title('LBP')

axs[1,4].imshow(cv2.cvtColor(after_rotated, cv2.COLOR_BGR2RGB))
axs[1,4].set_title('After Rotated Rect')


# Remove axes for cleaner look
for ax in axs.flat:
    ax.axis('off')

plt.tight_layout()
plt.show()

####Experiment with re-aligning the images first.

#Re-align images

