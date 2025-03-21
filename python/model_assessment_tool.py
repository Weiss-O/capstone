#This script will run through the datasets and assess the performance of the detector
from typing import List, Tuple
import os
import sys
import Detector as DE
import Classifier as CL
import ProposalGenerator as PG
import numpy as np
import cv2 as cv
import yaml
import datetime

#globals for baseline and test
baseline = None
image = None
timeString = datetime.datetime.now().strftime("%H:%M:%S")
iouThreshold_detections = 0.1
#Create Detector Object
testPredictor = CL.SAM2Predictor()

def create_detector(baseline_image):
    baselinePredictor = CL.SAM2Predictor()

    classifier = CL.IOUSegmentationClassifier(baseline_predictor=baselinePredictor,
                                                        test_predictor=testPredictor,
                                                        iou_threshold = 0.5,
                                                        baseline=cv.cvtColor(baseline_image, cv.COLOR_BGR2RGB))

    proposal_generator = PG.SSIMProposalGenerator(baseline=cv.cvtColor(baseline_image, cv.COLOR_BGR2RGB),
                                                    areaThreshold=8000)
    detector = DE.BasicDetector(baseline=baseline_image,
                                        proposal_generator=proposal_generator,
                                        classifier=classifier)
    return detector


def evaluate(detections: List[List[int]], baseline_annotations: List[List[int]], test_annotations: List[List[int]], baseline_img, test_img, M=None) -> Tuple[int, int]:
    """
    Evaluate true and false positives based on bounding box associations.
    
    Args:
        detections: List of detected bounding boxes.
        baseline_annotations: List of bounding boxes in the baseline image.
        test_annotations: List of bounding boxes in the test image.
    
    Returns:
        A tuple (true_positives, false_positives) where:
            - true_positives: Count of detections correctly identifying new or old objects.
            - false_positives: Count of detections mistakenly matching consistent objects or failing to match anything.
    """
    if M is not None:
        test_annotations = warp_bboxes(test_annotations, M)
    
    baseline_to_test = associate_bboxes(baseline_annotations, test_annotations)
    test_to_baseline = {t: b for b, t in baseline_to_test}
    
    new_objects = set(range(len(test_annotations))) - set(test_to_baseline.keys())
    consistent_objects = set(test_to_baseline.keys())
    old_objects = set(range(len(baseline_annotations))) - set(test_to_baseline.values())
    
    tp, fp, fn = 0, 0, 0
    
    detected_new_objects = set()
    
    for detection in detections:
        best_match = -1
        best_iou = iouThreshold_detections  # Same IoU threshold
        match_type = None
        
        for obj_set, label in [(new_objects, "new"), (consistent_objects, "consistent"), (old_objects, "old")]:
            for obj in obj_set:
                iou_val = iou(detection, test_annotations[obj])
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_match = obj
                    match_type = label
        
        if best_match != -1:
            if match_type == "new":
                tp += 1
                detected_new_objects.add(best_match)
            elif match_type == "old":
                tp += 1
            elif match_type == "consistent":
                fp += 1
        else:
            fp += 1  # Detection didn't match anything
    
    fn = len(new_objects - detected_new_objects)  # New objects that were not detected
    
    #visualize the bboxes. green for new, red for old, blue for consistent. yellow are baseline bboxes, white are detection bboxes

    # for i, bbox in enumerate(baseline_annotations):
    #     color = (0, 255, 255)
    #     if i in test_to_baseline.values():
    #         color = (255, 0, 0)
    #     cv.rectangle(baseline_img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color, 10)

    # for i, bbox in enumerate(test_annotations):
    #     color = (255, 255, 255)
    #     if i in test_to_baseline.keys():
    #         color = (255, 0, 0)
    #     elif i in new_objects:
    #         color = (0, 255, 0)
    #     elif i in consistent_objects:
    #         color = (255, 0, 0)
    #     cv.rectangle(test_img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color, 10)

    # #visualize the detections
    # for bbox in detections:
    #     cv.rectangle(test_img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (255, 255, 0), 10)

    # # create windows to display images, resized to 900px wide
    # cv.namedWindow("Baseline", cv.WINDOW_NORMAL)
    # cv.resizeWindow("Baseline", 900, 900)
    # cv.imshow("Baseline", baseline_img)
    # cv.namedWindow("Test", cv.WINDOW_NORMAL)
    # cv.resizeWindow("Test", 900, 900)
    # cv.imshow("Test", test_img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    return tp, fp, fn

def iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)
    
    inter_area = max(0, xb - xa) * max(0, yb - ya)
    box1_area = w1 * h1
    box2_area = w2 * h2
    
    iou_value = inter_area / float(box1_area + box2_area - inter_area)
    return iou_value

def associate_bboxes(boxes1: List[List[int]], boxes2: List[List[int]], iou_threshold=0.3) -> List[Tuple[int, int]]:
    """
    Associate bounding boxes between two lists based on IoU.
    
    Args:
        boxes1: List of bounding boxes in the first set (e.g., baseline or detections).
        boxes2: List of bounding boxes in the second set (e.g., test annotations).
        iou_threshold: Minimum IoU required to consider two boxes as a match.
    
    Returns:
        A list of tuples where each tuple (i, j) represents an association between
        a box in `boxes1` and a box in `boxes2`.
    """
    associations = []
    matched2 = set()
    
    for i, box1 in enumerate(boxes1):
        best_match = -1
        best_iou = iou_threshold
        
        for j, box2 in enumerate(boxes2):
            if j in matched2:
                continue
            
            iou_val = iou(box1, box2)
            if iou_val > best_iou:
                best_iou = iou_val
                best_match = j
        
        if best_match != -1:
            associations.append((i, best_match))
            matched2.add(best_match)
    
    return associations


root_path = "python/scans/"

#For folder in root_path
def export_results(scene, position, filename, tp, fp, fn):
    """
    Export detection evaluation results in a structured format.
    Saves results in CSV format for easy analysis.
    """
    output_file = f"detection_results_{timeString}.csv"
    file_exists = os.path.isfile(output_file)
    
    with open(output_file, "a") as f:
        if not file_exists:
            f.write("Scene,Position,File,True Positives,False Positives,False Negatives\n")
        f.write(f"{scene},{position},{filename},{tp},{fp},{fn}\n")


def warp_bboxes(bboxes, M):
    """
    Warps bounding boxes using the given transformation matrix M.
    :param bboxes: List of bounding boxes [(x, y, w, h), ...]
    :param M: Transformation matrix (2x3 for affine, 3x3 for homography)
    :return: Warped bounding boxes [(x', y', w', h'), ...]
    """
    warped_bboxes = []
    for (x, y, w, h) in bboxes:
        # Define the four corner points of the bounding box
        box_pts = np.array([
            [x, y], 
            [x + w, y], 
            [x, y + h], 
            [x + w, y + h]
        ], dtype=np.float32)

        # Reshape for transformation
        box_pts = np.expand_dims(box_pts, axis=1)

        # Apply transformation
        if M.shape == (2, 3):  # Affine transformation
            warped_pts = cv.transform(box_pts, M)
        else:  # Homography
            warped_pts = cv.perspectiveTransform(box_pts, M)

        # Extract min and max coords
        x_min, y_min = np.min(warped_pts, axis=0).flatten()
        x_max, y_max = np.max(warped_pts, axis=0).flatten()
        
        # Get new bounding box
        warped_bboxes.append((int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)))

    return warped_bboxes

def compute_transformation(baseline, test_image):
    """
    Compute the transformation matrix to align the test image with the baseline image.
    :return: Affine transformation matrix (2x3) or Homography matrix (3x3)
    """
    orb = cv.ORB_create()
    kp1, des1 = orb.detectAndCompute(baseline, None)
    kp2, des2 = orb.detectAndCompute(test_image, None)

    # Match keypoints
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) < 4:
        return None  # Not enough matches, return identity transformation

    src_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Estimate transformation
    M, mask = cv.estimateAffinePartial2D(src_pts, dst_pts, method=cv.RANSAC)

    return M



root_path = "python/scans/"

for scene in os.listdir(root_path):
    for position in os.listdir(os.path.join(root_path, scene)):
        with open(os.path.join(root_path, scene, position, "dataset.yaml")) as f:
            dataset = yaml.load(f, Loader=yaml.FullLoader)
        baseline = cv.imread(dataset["images"][0]["photo_path"])
        detector = create_detector(baseline)
        for file in dataset["images"]:
            filepath = file["photo_path"]
            print("Processing: " + filepath)
            # Load the image
            image = cv.imread(filepath)

             # Compute transformation matrix
            try:
                M = compute_transformation(baseline, image)
            except Exception as e:
                print(f"Error computing transformation: {e}")
                M = None
            # Detect the proposals
            detections = detector.detect(image)
            detections = [det.get_as_array() for det in detections]
            # Evaluate the detections
            TP, FP, FN = evaluate(detections, dataset["images"][0]["annotations"], file["annotations"], baseline, image, M)
            # Export the results in a structured format
            export_results(scene, position, filepath, TP, FP, FN)

print("Detection evaluation complete.")
print("Results exported to detection_results.csv.")

import pandas as pd
df = pd.read_csv(f"detection_results_{timeString}.csv")
summary = df.groupby(["Scene", "Position"])[["True Positives", "False Positives", "False Negatives"]].sum()
summary["TP/FP Ratio"] = summary.apply(lambda row: row["True Positives"] / row["False Positives"] if row["False Positives"] > 0 else np.inf, axis=1)
summary["Precision"] = summary["True Positives"] / (summary["True Positives"] + summary["False Positives"])

#Save the summary to a file
summary.to_csv(f"detection_summary_{timeString}.csv")