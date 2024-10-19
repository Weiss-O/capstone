import cv2
import numpy as np

def baseline_comp(img, baseline, threshold=30):
    # Convert images to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_baseline = cv2.cvtColor(baseline, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and improve differencing
    blurred_img = cv2.GaussianBlur(img, (5, 5), 0)
    blurred_baseline = cv2.GaussianBlur(baseline, (5, 5), 0)

    # Calculate the absolute difference between the blurred images
    diff = cv2.absdiff(blurred_img, blurred_baseline)

    #diff_rms is the rms of the three channels for each pixel in diff
    diff_rms = np.round(np.sqrt(np.sum(diff**2, axis=2))).astype(np.uint8)

    # Apply a fixed threshold to create a binary mask of the differences
    _, mask = cv2.threshold(diff_rms, threshold, 255, cv2.THRESH_BINARY)

    #Apply techniques to denoise and close the mask
    print(mask.shape)
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


    # Visualize the mask to check how the differences are being captured
    cv2.imshow('Mask', mask)

    # Optionally, apply morphological operations to clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Apply the mask to the original image to create a differenced image
    diff_img = img.copy()
    diff_img[mask == 255] = 255

    return diff_img


#Alternate differencing function:
def baseline_comp_alt(img, baseline):
    import cv2

    # load images
    image1 = baseline
    image2 = img

    # compute difference
    difference = cv2.subtract(image1, image2)

    # color the mask red
    Conv_hsv_Gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(Conv_hsv_Gray, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
    difference[mask != 255] = [0, 0, 255]

    return difference

# Load the images
baseline = cv2.imread('test_images/baseline_1.jpg')

#Load each image
imgs = []
imgs.append(cv2.imread('test_images/Mug.jpg'))
imgs.append(cv2.imread('test_images/mug_container.jpg'))
imgs.append(cv2.imread('test_images/mug_containter_pencil.jpg'))

# Compare each image to the baseline, have an adjustable threshold passed to the function
for img in imgs:
    difference = baseline_comp_alt(img, baseline)
    cv2.imshow('Difference Image', difference)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #Show the original image with the difference overlayed
    cv2.imshow('Overlay', cv2.addWeighted(img, 0.7, difference, 0.3, 0))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
