import os
import datetime
import cv2
import numpy as np
from skimage.metrics import structural_similarity
from skimage import feature
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Define utility functions
def dilation(threshed_img, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(threshed_img, kernel, iterations=1)

def erosion(threshed_img, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(threshed_img, kernel, iterations=1)

def fill(threshed_img):
    contours, _ = cv2.findContours(threshed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(threshed_img.shape, dtype=np.uint8)
    for contour in contours:
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return cv2.bitwise_or(threshed_img, closed_mask)

def lbp_map(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return feature.local_binary_pattern(gray, 1, 1, method="uniform")

# Load and process images
before = cv2.imread('./python/test_set/capture_2.jpg')
after = cv2.imread('./python/test_set/capture_41.jpg')
before_gray, after_gray = [cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0) for img in (before, after)]

# Compute SSIM
score, diff = structural_similarity(before_gray, after_gray, full=True)
print("Image Similarity: {:.4f}%".format(score * 100))
diff = (diff * 255).astype("uint8")
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

# Post-process
threshed_img_dilated = dilation(threshed_img.copy(), kernel_size=10)
lbp_img = lbp_map(before)
lbp_img = (lbp_img * 255).astype("uint8")
lbp_img = cv2.merge([lbp_img, lbp_img, lbp_img])

threshed_img = erosion(threshed_img, kernel_size=2)

# Contour analysis and visualization
contours, _ = cv2.findContours(threshed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
mask = np.zeros(before.shape, dtype='uint8')
filled_after = after.copy()

#Find evenly spaced points that lie on the contour
def get_contour_points(contour, num_points=3):
    contour = contour.squeeze()
    M = cv2.moments(contour)
    if M["m00"] != 0:
        centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    else:
        centroid = np.mean(contour, axis=0).astype(int)

    distances = np.linalg.norm(contour - centroid, axis=1)
    sorted_indices = np.argsort(distances)
    num_points = min(num_points, len(contour))
    selected_indices = sorted_indices[:num_points]
    return contour[selected_indices]

#function to find centroid of contour
def get_centroid(contour):
    M = cv2.moments(contour)
    if M["m00"] != 0:
        centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    else:
        centroid = np.mean(contour, axis=0).astype(int)
    return centroid

#function that finds centroid using above, but checks if it is within the contour, if not, finds the nearest pixel that is
def get_centroid_safe(contour):
    centroid = get_centroid(contour)
    if cv2.pointPolygonTest(contour, centroid, False) < 0:
        contour_points = get_contour_points(contour, num_points=5)
        distances = np.linalg.norm(contour_points - centroid, axis=1)
        centroid = contour_points[np.argmin(distances)]
    return centroid

for c in contours:
    if cv2.contourArea(c) > 400:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(before, (x, y), (x + w, y + h), (36,255,12), 2)
        cv2.rectangle(after, (x, y), (x + w, y + h), (36,255,12), 2)
        cv2.drawContours(mask, [c], 0, (255,255,255), -1)
        overlay = filled_after.copy()
        cv2.drawContours(overlay, [c], 0, (0,255,0), -1)
        cv2.addWeighted(overlay, 0.3, filled_after, 0.7, 0, filled_after)
        overlay = lbp_img.copy()
        cv2.drawContours(overlay, [c], 0, (0,0,255), -1)
        cv2.addWeighted(overlay, 0.3, lbp_img, 0.7, 0, lbp_img)
        point1 = get_centroid_safe(c)
        cv2.circle(filled_after, tuple(point1), 1, (255, 0, 0), -1)
        point2 = get_centroid(c)
        cv2.circle(filled_after, tuple(point2), 1, (0, 0, 255), -1)


# Plot images as subplots
fig, axs = plt.subplots(2, 4, figsize=(15, 8))
images = [
    (before, "Before Image"), (after, "After Image"),
    (diff, "Difference Image"), (threshed_img, "Thresholded Diff"),
    (threshed_img_dilated, "Dilated Threshold"), (mask, "Contour Mask"),
    (filled_after, "Filled After Image"), (lbp_img, "LBP Map with Contours")
]

for ax, (img, title) in zip(axs.ravel(), images):
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else img, cmap='gray')
    ax.set_title(title)
    ax.axis('off')

plt.tight_layout()
plt.show(block=True)

# Saving images to output directory
output_dir = os.path.join('output', datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
os.makedirs(output_dir, exist_ok=True)
for img, name in zip([before, after, diff, threshed_img, threshed_img_dilated, mask, filled_after, lbp_img], 
                     ["before.jpg", "after.jpg", "diff.jpg", "threshed.jpg", "dilated.jpg", "mask.jpg", "filled_after.jpg", "lbp.jpg"]):
    cv2.imwrite(os.path.join(output_dir, name), img)

print(f"Images saved to {output_dir}")
