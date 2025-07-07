import cv2
import numpy as np
import yaml

with open("config.yaml") as file:
    config = yaml.safe_load(file)

class CameraReferenceFrame():
    def __init__(self, focal_length, resX, resY, radial_distance, pixel_size):
        self.focal_length = focal_length #Focal Length in mm
        self.resX = resX # Resolution in Pixels
        self.resY = resY # Resolution in Pixels
        self.pixel_size= pixel_size #Pixel size in mm
        self.radial_distance = radial_distance

        self.sensor_width = resX * pixel_size
        self.sensor_height = resY * pixel_size
        self.T_cam_photo = self.calculate_T_cam_photo()
        self.FOV = self.calculate_fov()
        self.calculate_T_cam_world(0, 0)

    #Function to project out a point into the 3D reference frame of the camera
    def calculate_T_cam_photo(self):
        return np.array([
            [self.focal_length/self.pixel_size, 0, self.resX/2],
            [0, self.focal_length/self.pixel_size, self.resY/2],
            [0, 0, 1]
        ])
    
    @classmethod
    def angle_between_rays(ray1, ray2, degrees=True):
        dot_product = np.dot(ray1, ray2)
        dot_product = np.clip(dot_product, -1.0, 1.0)  # Avoid numerical errors
        
        angle = np.arccos(dot_product)
        
        return np.degrees(angle) if degrees else angle
    
    @classmethod
    def spherical_to_cartesian(theta, phi, degrees=True):
        if degrees:
            theta = np.radians(theta)
            phi = np.radians(phi)
        
        x = np.cos(theta) * np.cos(phi)
        y = np.sin(theta) * np.cos(phi)
        z = np.sin(phi)
        
        return np.array([x, y, z])


    #Change from 3D point in camera reference frame to 2D point in image reference frame
    def cam_to_image(self, point):
        return np.dot(self.T_cam_photo, point[:3]/point[2])

    def image_to_cam(self, point, distance):
        return np.dot(np.linalg.inv(self.T_cam_photo), point)*distance
    
    def image_to_sensor_plane(self, point):
        return self.image_to_cam(point, self.focal_length)

    def calculate_fov(self):
        return [np.rad2deg(2*np.arctan(x/(2*self.focal_length))) for x in [
            self.sensor_width,
            self.sensor_height,
            np.sqrt(self.sensor_width**2 + self.sensor_height**2)
            ]]
    
    #Calculate matrix to transform camera coordinates to world coordinates (homogenous)
    def calculate_T_cam_world(self, theta, phi):
        theta_rad = np.radians(theta)
        phi_rad = np.radians(phi)

        T_cam_world = np.linalg.inv(np.array([
            [np.sin(theta_rad), -np.cos(theta_rad), 0, 0],
            [-np.sin(phi_rad) * np.cos(theta_rad), -np.sin(phi_rad) * np.sin(theta_rad), -np.cos(phi_rad), 0],
            [np.cos(phi_rad) * np.cos(theta_rad), np.cos(phi_rad) * np.sin(theta_rad), -np.sin(phi_rad), -self.radial_distance],
            [0, 0, 0, 1]
        ]))

        self.T_cam_world = T_cam_world
    
    def estimate_theta_phi_abs(self, point, assumed_dist=12*25.4*14.25):
        #Project the point out the distance
        cam_point = np.append(self.image_to_cam(point, assumed_dist - self.radial_distance), 1)
        #Project the point out to the world frame
        world_point = np.dot(self.T_cam_world, cam_point)
        #Calculate the angles
        phi = np.arcsin(-world_point[2]/np.linalg.norm(world_point[:3]))
        theta = np.arctan2(world_point[1], world_point[0])

        return theta, phi
    
referenceFrame = CameraReferenceFrame(4.74, 4608, 2592, 50, 1.4e-3)

#Start capturing camera feed from webcam
# Check for available webcams and use the first one found

cap = cv2.VideoCapture(0)
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
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        for i in range(len(ids)):
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.05, camera_matrix, dist_coeffs)
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs, tvecs, 0.03)

            theta, phi = referenceFrame.estimate_theta_phi_abs(tvecs[0][0])
            print(f"Marker ID: {ids[i][0]}: theta = {theta:.4f}, phi = {phi:.4f}")

    cv2.imshow("ArUco Tracker", frame)

    #Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()