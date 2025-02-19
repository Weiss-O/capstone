import numpy as np
import cv2

#Function to get a specified number of points spaced equally on a contour
def get_contour_points(contour, num_points = 3):
    contour = contour.squeeze()
    num_points = min(num_points, len(contour))
    indices = np.linspace(0, len(contour)-1, num_points, dtype=int)
    return contour[indices]

#TODO: Verify this function works as it should.
def get_centroid(contour):
    M = cv2.moments(contour)
    if M["m00"] != 0:
        centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    else:
        x_sum = sum(point[0][0] for point in contour)
        y_sum = sum(point[0][1] for point in contour)
        count = len(contour)
        centroid = (int(x_sum / count), int(y_sum / count))
    return centroid

#Function to find the centroud of a contour that is guaranteed to be inside the contour
def get_centroid_safe(contour):
    centroid = get_centroid(contour)
    if cv2.pointPolygonTest(contour, centroid, False) < 0:
        contour_points = get_contour_points(contour, num_points=10)
        distances = np.linalg.norm(contour_points - centroid, axis=1).argmin()
        min_dist_index = np.argmin(distances)
        centroid=(contour_points[min_dist_index][0], contour_points[min_dist_index][1])
    return centroid

#Function to calculate the intersection over union of two contours
def calculate_iou(contour1, contour2):
    intersection = np.sum(np.logical_and(contour1, contour2))
    union = np.sum(np.logical_or(contour1, contour2))
    iou = intersection / union if union != 0 else 0
    return iou