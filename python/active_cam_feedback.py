import cv2
import numpy as np
import Camera
import yaml

with open("config.yaml") as file:
    config = yaml.safe_load(file)

referenceFrame = Camera.CameraReferenceFrame(4.74, 4608, 2592, 50, 1.4e-3)

#Start capturing camera feed from webcam
# Check for available webcams and use the first one found

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()


# Define the ArUco dictionary and detector parameters
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

camera_matrix = referenceFrame.calculate_T_cam_photo()
dist_coeffs = np.zeros(5)
#Look for aruco markers in video feed, show coordinate frame representations and IDs.
while True:
    ret, frame = cap.read()
    if not ret:
        break

    #convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Detect the markers
    corners, ids = detector.detectMarkers(gray)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        for i in range(len(ids)):
            rvecs, tvecs = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.05, camera_matrix, dist_coeffs)
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs, tvecs, 0.03)

            theta, phi = referenceFrame.estimate_theta_phi_abs(tvecs[0][0])
            print(f"Marker ID: {ids[i[0]]:}: theta = {theta:.2f}, phi = {phi:.2f}")

    cv2.imshow("ArUco Tracker", frame)

    #Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()