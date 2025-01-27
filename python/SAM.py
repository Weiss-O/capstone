# Testing out using SAM2 with cuda in WSL
# Code is from the SAM2 example jupyter notebooks

#Necessary imports and helper functions for displaying points, boxes and masks
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import structural_similarity
import cv2

device = torch.device("cuda")
root_dir = os.path.dirname(os.path.abspath(__file__)) + "/"

if device.type == "cuda":
        #use bfloat16
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # Turn on tfloat32 for Ampere GPUs (Lenovo Legion has one)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

np.random.seed(3)

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()

image_path = os.path.join(root_dir, "test_images/3_obj_darker.jpg")
image = Image.open(image_path)
image = np.array(image.convert("RGB"))

plt.figure(figsize=(10,10))
plt.imshow(image)
plt.axis('on')
plt.show()

##Loading the SAM2 model and predictor
import sys

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

sam_path = os.path.expanduser("~/sam2")
# Load Sam checkpoint "~/sam2/checkpoints/sam2.1_hiera_large.pt"
sam2_checkpoint = os.path.join(sam_path, "checkpoints/sam2.1_hiera_large.pt")
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)


predictor = SAM2ImagePredictor(sam2_model)

#Next, we process the image to produce an "image embedding"
#Once generated, SAM2 remembers this embedding and will use it for later sam prediction
predictor.set_image(image)

#Point input to model
#Prompt sam with np array of input x, y points.
#labels:
# 0 - background (exclude)
# 1 - foreground (include)

input_point = np.array([[1100, 600]])
input_label = np.array([1])

plt.figure(figsize=(10, 10))
plt.imshow(image)
show_points(input_point, input_label, plt.gca())
plt.axis('on')
plt.show()

print(predictor._features["image_embed"].shape, predictor._features["image_embed"][-1].shape)

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True
)
sorted_ind = np.argsort(scores)[::-1] #sort in descending order of scores
masks = masks[sorted_ind]
scores = scores[sorted_ind]
logits = logits[sorted_ind]

print(masks.shape) # (Number of Masks) x H x W

show_masks(image, masks, scores, point_coords=input_point, input_labels=input_label, borders=True)

best_mask = masks[np.argmax(scores)] #mask is in the form of a binary image
mask_img = np.zeros_like(image)
mask_img[best_mask == 1] = image[best_mask == 1] 

plt.figure(figsize=(10, 10))
plt.imshow(mask_img)
plt.axis("on")
plt.show()

#Specifying a specific object with additional points
#**If available, a mask from a previous iteration can also be supplied to the model to aid in prediction
# Request a single mask by setting multimask_output = False

input_point = np.array([[500, 900],
                        [750, 800],
                        [1000, 900],
                        [1250, 800],
                        [1500, 900]
                       ])

input_label = np.array([1,1,1,1,1])

mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask

masks, scores, _ = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=False,
)

show_masks(image, masks, scores, point_coords=input_point, input_labels=input_label)

#Trying out SSIM > SAM pipeline

baseline_path = "output/baseline.jpg"
test_path = "output/test.jpg"

baseline_sam = Image.open(baseline_path)
test_sam = Image.open(test_path)

baseline_sam = np.array(baseline_sam.convert("RGB"))
test_sam = np.array(test_sam.convert("RGB"))


predictor.set_image(baseline_sam)

predictor2 = SAM2ImagePredictor(sam2_model)

predictor2.set_image(test_sam)

baseline = cv2.imread(baseline_path)
test = cv2.imread(test_path)
baseline, test = [cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0) for img in (baseline, test)]

def erosion(threshed_img, kernel_size=3):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(threshed_img, kernel, iterations=1)

#compute SSIM
score, diff = structural_similarity(baseline, test, full=True)
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

threshed_img = erosion(threshed_img, kernel_size=2)

# Contour analysis and visualization
contours, _ = cv2.findContours(threshed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
mask_visual = np.zeros((*baseline.shape, 3), dtype='uint8')

#Function(s) to generate point prompts from the contours
#Find evenly spaced points that lie on the contour
def get_contour_points(contour, num_points=3):
    contour = contour.squeeze()
    num_points = min(num_points, len(contour))
    indices = np.linspace(0, len(contour) - 1, num_points, dtype=int)
    return contour[indices]

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
        contour_points = get_contour_points(contour, num_points=10)
        distances = np.linalg.norm(contour_points - centroid, axis=1)
        centroid = contour_points[np.argmin(distances)]
    return centroid

def calculate_iou(pred_mask, gt_mask):
    """
    Calculate the Intersection over Union (IoU) between predicted and ground truth masks.
    
    Parameters:
    - pred_mask (numpy.ndarray): The predicted binary mask (same shape as gt_mask).
    - gt_mask (numpy.ndarray): The ground truth binary mask (same shape as pred_mask).
    
    Returns:
    - float: The IoU score between the predicted and ground truth masks.
    """
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)
    
    intersection = np.sum(np.logical_and(pred_mask, gt_mask))  # Intersection of the two masks
    union = np.sum(np.logical_or(pred_mask, gt_mask))  # Union of the two masks
    
    iou = intersection / union if union != 0 else 0.0  # Return IoU, or 0 if no union
    return iou

for c in contours:
    if cv2.contourArea(c) > 400:
        point = np.array([get_centroid_safe(c)])
        label = np.array([1])

        masks_baseline, scores_baseline, logits_baseline = predictor.predict(
            point_coords=point,
            point_labels=label,
            multimask_output=True
        )

        masks, scores, logits = predictor2.predict(
            point_coords=point,
            point_labels=label,
            multimask_output=True
        )

        # Find the highest confidence mask for each predictor (highest score)
        best_mask_baseline = masks_baseline[np.argmax(scores_baseline)]  # Best mask from baseline predictor
        best_mask_test = masks[np.argmax(scores)]  # Best mask from test predictor

        iou = calculate_iou(best_mask_baseline, best_mask_test)

        cv2.drawContours(mask_visual, [c], 0, (255,255,255), -1)
        threshold = 0.5
        text_color = (0, 0, 255)
        if iou<threshold:
            text_color = (0, 255, 0)

        cv2.putText(
            mask_visual,
            f"IoU: {iou:.4f}",
            (point[0][0], point[0][1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,  # Font scale
            text_color,  # Red text
            1,  # Thickness
            cv2.LINE_AA
        )

        #Add code here to also add the contour being tested to the image
        contour_img = test_sam.copy()
        cv2.drawContours(contour_img, [c], -1, (0, 255, 0), 2)

        plt.figure(figsize=(10, 10))
        plt.imshow(contour_img)
        plt.axis("on")

        # Show the prompt point
        plt.scatter(point[0][0], point[0][1], color='red', s=50, label='Prompt Point')


        show_mask(best_mask_baseline, plt.gca(), borders=True)
        show_mask(best_mask_test, plt.gca(), borders=True)

        #Display IOU in top right

        plt.text(
            0.95, 0.95, f"IoU: {iou:.4f}",  # Position the text in the top-right corner
            color='white', fontsize=12, ha='right', va='top', transform=plt.gca().transAxes
        )

        plt.show()


mask_visual=cv2.cvtColor(mask_visual, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10,10))
plt.imshow(mask_visual)
plt.axis("on")
plt.show()