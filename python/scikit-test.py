from skimage.metrics import structural_similarity
import cv2
import numpy as np

# Load images
before = cv2.imread('test_images/fr_baseline.jpg')
after = cv2.imread('test_images/fr_test.jpg')

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
else:
    threshed_img = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV)[1]



contours = cv2.findContours(threshed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]

mask = np.zeros(before.shape, dtype='uint8')
filled_after = after.copy()

for c in contours:
    area = cv2.contourArea(c)
    if area > 400:
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(before, (x, y), (x + w, y + h), (36,255,12), 2)
        cv2.rectangle(after, (x, y), (x + w, y + h), (36,255,12), 2)
        cv2.rectangle(diff_box, (x, y), (x + w, y + h), (36,255,12), 2)
        cv2.drawContours(mask, [c], 0, (255,255,255), -1)
        cv2.drawContours(filled_after, [c], 0, (0,255,0), -1)
        
# # Resize images for display
# before_resized = cv2.resize(before, (600, 400))
# after_resized = cv2.resize(after, (600, 400))
# diff_resized = cv2.resize(diff, (600, 400))
# diff_box_resized = cv2.resize(diff_box, (600, 400))
# mask_resized = cv2.resize(mask, (600, 400))
# filled_after_resized = cv2.resize(filled_after, (600, 400))

# Display images
cv2.imshow('before', before)
cv2.imshow('after', after)
cv2.imshow('diff', diff)
cv2.imshow('diff_box', diff_box)
cv2.imshow('mask', mask)
cv2.imshow('filled after', filled_after)
cv2.waitKey()

#save images to output folder, under new subfolder named based on the current date and time
import os
import datetime

# Get the current date and time
now = datetime.datetime.now()
date_time = now.strftime("%Y-%m-%d %H-%M-%S")

# Create a new directory to save the output images
output_dir = 'output/' + date_time
os.makedirs(output_dir)

# Save the images to the output directory
cv2.imwrite(output_dir + '/before.jpg', before)
cv2.imwrite(output_dir + '/after.jpg', after)
cv2.imwrite(output_dir + '/diff.jpg', diff)
cv2.imwrite(output_dir + '/diff_box.jpg', diff_box)
cv2.imwrite(output_dir + '/mask.jpg', mask)
cv2.imwrite(output_dir + '/filled_after.jpg', filled_after)
