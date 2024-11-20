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
before = cv2.imread('python/test_set/capture_2.jpg')
after = cv2.imread('python/test_set/capture_41.jpg')


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

for c in contours:
    area = cv2.contourArea(c)
    if area > 40:
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
cv2.imshow('before', before)
cv2.imshow('after', after)
cv2.imshow('diff', diff)
cv2.imshow('diff_box', diff_box)
cv2.imshow('mask', mask)
cv2.imshow('mask_dil', mask_dil)
cv2.imshow('filled after', filled_after)
cv2.imshow('lbp', lbp_img)
cv2.waitKey()


# Function to update the histogram with the given bin and area filter
def update(val):
    # Get the number of bins and the minimum area from the sliders
    bins = int(slider_bins.val)
    min_area = slider_area.val
    
    # Filter contour areas based on the minimum area
    filtered_areas = [area for area in contour_areas if area >= min_area]
    
    # Clear previous histogram and plot the new one
    ax.clear()

    # Logarithmic binning
    log_bins = np.logspace(np.log10(min(filtered_areas)), np.log10(max(filtered_areas)), bins)
    
    ax.hist(filtered_areas, bins=log_bins, color='blue', edgecolor='black')
    ax.set_title('Histogram of Contour Areas')
    ax.set_xlabel('Area')
    ax.set_ylabel('Frequency')
    ax.set_xscale('log')  # Set the x-axis to a logarithmic scale
    fig.canvas.draw_idle()

# Create the figure and axis for plotting
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)

# Initial histogram plot with logarithmic bins
ax.hist(contour_areas, bins=np.logspace(np.log10(min(contour_areas)), np.log10(max(contour_areas)), 20), color='blue', edgecolor='black')
ax.set_title('Histogram of Contour Areas')
ax.set_xlabel('Area')
ax.set_ylabel('Frequency')
ax.set_xscale('log')  # Set the x-axis to a logarithmic scale

# Slider for adjusting the number of bins
ax_bins = plt.axes([0.2, 0.01, 0.65, 0.03], facecolor='lightgoldenrodyellow')
slider_bins = Slider(ax_bins, 'Bins', 1, 100, valinit=20, valstep=1)

# Slider for adjusting the minimum area
ax_area = plt.axes([0.2, 0.06, 0.65, 0.03], facecolor='lightgoldenrodyellow')
slider_area = Slider(ax_area, 'Min Area', 0, 1000, valinit=0, valstep=1)

# Attach the update function to the sliders
slider_bins.on_changed(update)
slider_area.on_changed(update)

plt.show()

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
# cv2.imwrite(output_dir + '/before.jpg', before)
# cv2.imwrite(output_dir + '/after.jpg', after)
# cv2.imwrite(output_dir + '/diff.jpg', diff)
# cv2.imwrite(output_dir + '/diff_box.jpg', diff_box)
cv2.imwrite(output_dir + '/mask.jpg', mask)
# cv2.imwrite(output_dir + '/filled_after.jpg', filled_after)
