#Testing to see what edge detection looks like for baseline images and whether it could reasonably be used to filter baseline_comp results
#Source 1: https://learnopencv.com/edge-detection-using-opencv/

import cv2
import numpy as np

def sobel_edge_detection(image):
    # Ensure image is grayscale
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # Define Sobel kernels for x and y directions
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]])

    # Apply the Sobel filter in x and y directions using convolution
    grad_x = cv2.filter2D(gray_image, cv2.CV_64F, sobel_x)
    grad_y = cv2.filter2D(gray_image, cv2.CV_64F, sobel_y)

    # Calculate the magnitude of the gradient
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Normalize to 0-255 and convert to uint8
    grad_magnitude = np.uint8(255 * grad_magnitude / np.max(grad_magnitude))

    return grad_magnitude


# Read the original image
img = cv2.imread('python/test_set/capture_2.jpg') 
# Display original image
cv2.imshow('Original', img)
cv2.waitKey(0)
 
# Convert to graycsale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Blur the image for better edge detection
img_blur = cv2.GaussianBlur(img_gray, (5,5), 0) 
 
# Sobel Edge Detection
sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection
sobelMag = sobel_edge_detection(img_blur)

denoised_sobelMag_fastNl = cv2.fastNlMeansDenoising(sobelMag, None, 10, 7, 21)

_, threshed_img = cv2.threshold(sobelMag, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
_, threshed_img_denoised = cv2.threshold(denoised_sobelMag_fastNl, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

# Display Sobel Edge Detection Images
# cv2.imshow('Sobel Magnitude', sobelMag)
# cv2.waitKey(0)
# cv2.imshow('Thresholded Sobel Magnitude', threshed_img)
# cv2.waitKey(0)
# cv2.imshow('Denoised Sobel Magnitude', denoised_sobelMag_fastNl)
# cv2.waitKey(0)
cv2.imshow('Thresholded Denoised Sobel Magnitude', threshed_img_denoised)
cv2.waitKey(0)

diff_img = cv2.imread(r'python\output\2024-11-13_20-19-16\threshed.jpg')

if len(diff_img.shape) == 2:
    diff_img = cv2.cvtColor(diff_img, cv2.COLOR_GRAY2BGR)

overlay = diff_img.copy()
overlay[threshed_img_denoised == 0] = [0, 0, 255]
#Overlay threshed_img_denoised but in red on top of diff_img


output_image = cv2.addWeighted(overlay, 1, diff_img, 0, 0)



cv2.imshow('unfiltered detections', diff_img)
cv2.waitKey(0)
cv2.imshow('filtered detections', output_image)
cv2.waitKey(0)

# Canny Edge Detection
edges = cv2.Canny(image=img_blur, threshold1=50, threshold2=50) # Canny Edge Detection
# Display Canny Edge Detection Image
cv2.imshow('Canny Edge Detection', edges)
cv2.waitKey(0)
 
cv2.destroyAllWindows()