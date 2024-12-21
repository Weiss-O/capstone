from __future__ import print_function
import cv2 as cv
import glob
import re

from skimage.metrics import structural_similarity
import numpy as np

# Define the folder path containing images
folderpath = r"C:\Users\user\OneDrive - University of Waterloo\Documents\Python Scripts\capstone\python\test_set\capture_*.jpg"  # Specify the correct path to your image files

# Create a list of image file paths
vectorOfimages = glob.glob(folderpath)  # This will be a list of file paths matching the pattern

# Sort the images if they need to be processed in a specific order
def extract_number(filepath):
    match = re.search(r'(\d+)', filepath)
    return int(match.group(1)) if match else -1

vectorOfimages.sort(key=extract_number)  # Sort based on the number in the filename


# Function to compute the structural similarity index between two images
def image_diff(before, after):
    # Convert images to grayscale
    before_gray = cv.cvtColor(before, cv.COLOR_BGR2GRAY)
    after_gray = cv.cvtColor(after, cv.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and improve differencing
    before_gray = cv.GaussianBlur(before_gray, (5, 5), 0)
    after_gray = cv.GaussianBlur(after_gray, (5, 5), 0)

    # Compute SSIM between the two images
    (score, diff) = structural_similarity(before_gray, after_gray, full=True)
    print("Image Similarity: {:.4f}%".format(score * 100))

    # The diff image contains the actual image differences between the two images
    # and is represented as a floating point data type in the range [0,1] 
    # so we must convert the array to 8-bit unsigned integers in the range
    # [0,255] before we can use it with OpenCV
    diff = (diff * 255).astype("uint8")
    diff_box = cv.merge([diff, diff, diff])

    thresh, threshed_img = cv.threshold(diff, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)

    # Check how close the diff values are to the threshold on average
    mean_diff_value = np.mean(np.abs(diff-thresh))
    print("Mean diff value: {:.2f}".format(mean_diff_value))
    print("Otsu's threshold value: {:.2f}".format(thresh))

    # Determine if Otsu's thresholding is appropriate
    if mean_diff_value > 10:  # You can adjust this threshold value as needed
        print("Otsu's thresholding is appropriate.")
    else:
        threshed_img = cv.threshold(diff, 0, 255, cv.THRESH_BINARY_INV)[1]


    contours = cv.findContours(threshed_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    return score, contours


# Choose background subtraction method
algo = 'MOG2'  # Options: 'MOG2' or 'KNN'

# Initialize background subtractor
if algo == 'MOG2':
    backSub = cv.createBackgroundSubtractorMOG2()
else:
    backSub = cv.createBackgroundSubtractorKNN()


baseline = cv.imread(vectorOfimages[0])
# Process each image in the vectorOfimages
for i in range(1, len(vectorOfimages)):
    # Read each image file
    image = cv.imread(vectorOfimages[i])

    # Check if the image was successfully loaded
    if image is None:
        print(f"Error loading image {vectorOfimages[i]}")
        continue

    # Apply background subtraction
    fgMask = backSub.apply(image)
    
    #Use SSIM for image differencing
    score, contours = image_diff(baseline, image)
    
    #Create a copy of the image with brightness reduced by half
    ssim_result = image.copy()
    ssim_result = cv.addWeighted(ssim_result, 0.5, ssim_result, 0, 0)

    for c in contours:
        if cv.contourArea(c) > 0:
            cv.drawContours(ssim_result, [c], 0, (0, 0, 255), -1)  # Red color
            overlay = ssim_result.copy()
            cv.addWeighted(overlay, 0.5, ssim_result, 0.5, 0, ssim_result)  # Make it slightly translucent
            
    #Print similarity score in top left corner
    cv.putText(ssim_result, "Similarity: {:.2f}%".format(score * 100), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display results (optional)
    cv.imshow('Foreground Mask', fgMask)
    cv.imshow('SSIM Result', ssim_result)
    cv.imshow('Original Image', image)

    while True:
        # Wait for 'q' or ESC to exit
        keyboard = cv.waitKey(30)
        if keyboard == ord('q') or keyboard == 27:
            break

# Release all resources
cv.destroyAllWindows()