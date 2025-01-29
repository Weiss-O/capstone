#File for all camera-related functionality
from abc import ABC, abstractmethod

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
        return None #TODO: This should match the return type of the actual capture function
    
    def getDistortionMatrix(self):
        return self.camera_settings["distortion_matrix"]

class PiCamera():
    def __init__(self, camera_settings):
        from picamera2 import Picamera2

        self.camera_settings = camera_settings
        self.picam2 = Picamera2()
        self.initialize()

    def initialize(self):
        from libcamera import controls
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
        self.picam2.capture_file(image_path)

    def capture(self):
        return self.picam2.capture()

    def getDistortionMatrix(self):
        return self.camera_settings["distortion_matrix"]