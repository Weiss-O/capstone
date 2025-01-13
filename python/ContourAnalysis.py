import numpy as np
import cv2

def get_contour_points(contour, num_points = 3):
    contour = contour.squeeze()
    num_points = min(num_points, len(contour))
    indices = np.linspace(0, len(contour)-1, num_points, dtype=int)
    return contour[indices]

def get_centroid(contour):
    M = cv2.moments(contour)
    if M["m00"] != 0:
        centroid = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
    else:
        centroid = np.mean(contour, axis=0).astype(int)
    return centroid

def get_centroid_safe(contour):
    centroid = get_centroid(contour)
    if cv2.pointPolygonTest(contour, centroid, False) < 0:
        contour_points = get_contour_points(contour, num_points=10)
        distances = np.linalg.norm(contour_points - centroid, axis=1).argmin()
        centroid=contour_points[np.argmin(distances)]
    return centroid

def calculate_iou(contour1, contour2):
    intersection = np.sum(np.logical_and(contour1, contour2))
    union = np.sum(np.logical_or(contour1, contour2))
    iou = intersection / union if union != 0 else 0
    return iou