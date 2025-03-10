#File for all camera-related functionality
from abc import ABC, abstractmethod
import numpy as np
import cv2
import os
import Server

if os.environ.get('RPI', 'False').lower() == 'true':
    from picamera2 import Picamera2 #type:ignore
    from libcamera import controls #type:ignore

class Camera(ABC):
    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def stop(self):
        pass

    @abstractmethod
    def capture_file(self, image_path):
        pass

    @abstractmethod
    def capture(self):
        pass

    @abstractmethod
    def getDistortionMatrix(self):
        pass

    @abstractmethod
    def initialize(self):
        pass

class CameraStandIn(Camera):
    def __init__(self, camera_settings):
        self.camera_settings = camera_settings
        self.reference_frame = CameraReferenceFrame(4.74, 4608, 2592, 50, 1.4e-3)
        self.initialize()

    def initialize(self):
        print(f"Set resolution to {self.camera_settings['resolution'][0]}x{self.camera_settings['resolution'][1]} px")
        print(f"Set lens position to {self.camera_settings['LensPosition']}")
    
    def start(self):
        print("Camera Stand In Started")
    
    def stop(self):
        print("Camera Stand In Stopped")
    
    def capture_file(self, image_path):
        print(f"Captured image to {image_path}")
    
    def capture(self):
        print("Captured image")
        image = cv2.imread("python/test_set/capture_42.jpg")
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image #TODO: This should match the return type of the actual capture function
    
    def update_ReferenceFrame(self, theta, phi):
        self.reference_frame.calculate_T_cam_world(theta, phi)

    def calculate_pointing_ray(self, point, degrees=True):
        pointing_coord = self.reference_frame.estimate_theta_phi_abs(point)
        if degrees:
            pointing_coord = np.rad2deg(pointing_coord)
        return self.reference_frame.estimate_theta_phi_abs(point)
    
    def getDistortionMatrix(self):
        return self.camera_settings["distortion_matrix"]

def ft_to_mm(ft):
    return ft*304.8

class PiCamera(Camera):
    def __init__(self, camera_settings):
        self.camera_settings = camera_settings
        self.picam2 = Picamera2()
        self.reference_frame = CameraReferenceFrame(4.74, 4608, 2592, 50, 1.4e-3)
        self.reference_frame.calculate_T_cam_world(45, 45)
        self.initialize()

    def initialize(self):
        
        # Configure camera for high resolution
        camera_config = self.picam2.create_still_configuration(main={"size": (self.camera_settings["resolution"][0], self.camera_settings["resolution"][1])})  # Max resolution for Pi Camera v2
        self.picam2.configure(camera_config)

        # Set the manual focus mode
        self.picam2.set_controls({"AfMode": controls.AfModeEnum.Manual, "LensPosition": self.camera_settings["LensPosition"]})  # Adjust LensPosition as needed

    def start(self):
        self.picam2.start(show_preview=False)

    def stop(self):
        self.picam2.stop()

    def capture_file(self, image_path):
        try:
            image = self.picam2.capture_array()
            image = image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(image_path, image)
        except Exception as e:
            return e

    def capture(self):
        try:
            image = self.picam2.capture_array()
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            return image
        except Exception as e:
            return e
    
    def update_ReferenceFrame(self, theta, phi):
        self.reference_frame.calculate_T_cam_world(theta, phi)

    def calculate_pointing_ray(self, point, degrees=True):
        pointing_coord = self.reference_frame.estimate_theta_phi_abs(point)
        if degrees:
            pointing_coord = np.rad2deg(pointing_coord)
        return pointing_coord

    def getDistortionMatrix(self):
        return self.camera_settings["distortion_matrix"]
    
    def capture_and_send_remote(self, server, image_name, pos=None): #TODO: There is probably a better place for this functionality
        Server.send_bytes(server, b'STORE_IMAGE')
        print("Sent STORE_IMAGE command")
        try:
            image = self.capture()
            print("Image Captured")
        except Exception as e:
            print(f"Error capturing image: {e}")
            return
        success, encoded_image = cv2.imencode('.jpg', image)
        print("Encoded Image")
        if not success:
            raise Exception("Error encoding image")
        image_bytes = encoded_image.tobytes()

        Server.send_bytes(server, image_bytes)
        print("Sent image")
        if pos is not None:
            Server.send_bytes(server, image_name.encode())
        else:
            Server.send_coords(server, pos)



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
    
    def estimate_theta_phi_abs(self, point, assumed_dist=ft_to_mm(14.25)):
        #Project the point out the distance
        cam_point = np.append(self.image_to_cam(point, assumed_dist - self.radial_distance), 1)
        #Project the point out to the world frame
        world_point = np.dot(self.T_cam_world, cam_point)
        #Calculate the angles
        phi = np.arcsin(-world_point[2]/np.linalg.norm(world_point[:3]))
        theta = np.arctan2(world_point[1], world_point[0])

        return theta, phi
    

if __name__ == "__main__":
    import yaml
    import socket
    import Server
    with open('config.yaml') as file:
        config = yaml.safe_load(file)
    print("File loaded")
    camera = PiCamera(config["camera_settings"])
    camera.start()
    print("Camera initialized")
    HOST = config["server_settings"]["HOST"]
    PORT = config["server_settings"]["PORT"]
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print("Server initialized")
    server.connect((HOST, PORT))
    print("Connected")
    count = 0
    while True:
        command = input("Enter command: ")
        if command == "C":
            camera.capture_and_send_remote(server, f"manual_test_{count}")
        elif command == b'X':
            break
        else:
            print(f"Unknown command: {command}")
            break