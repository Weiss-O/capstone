#!/usr/bin/env python3
import serial
import numpy as np
import glob
import os

class Controller():
    def __init__(self, controller_settings):
        self.settings = controller_settings
        ports = glob.glob('/dev/ttyACM*')
        if not ports:
            raise IOError("No ttyACM devices found")
        self.port = ports[0]
        self.ser = serial.Serial(self.port, self.settings["baudrate"], timeout = self.settings["timeout"])
        self.ser.reset_input_buffer()
        self.is_open = self.ser.is_open
        self.current_position = [0, 0]
        self.previous_signs = [1, 1]
        self.current_signs = [1, 1]
        
    #Expexts an input in degreees
    def point_camera(self, theta, phi):
        theta_cam = theta - self.settings["camera_offset"]["theta"]
        phi_cam = phi - self.settings["camera_offset"]["phi"]
        theta_steps = degrees_to_steps(theta_cam, self.settings["steps_per_revolution"])
        phi_steps = degrees_to_steps(phi_cam, self.settings["steps_per_revolution"])
        theta_actual = steps_to_degrees(theta_steps, self.settings["steps_per_revolution"]) + self.settings["camera_offset"]["theta"]
        phi_actual = steps_to_degrees(phi_steps, self.settings["steps_per_revolution"]) + self.settings["camera_offset"]["phi"]
        
        try: 
            self.moveAbsolute(theta_steps, phi_steps)
            return theta_actual, phi_actual
        except Exception as e:
            print(f"Error pointing camera: {e}")
            return e
    
    def point_projector(self, theta, phi):
        theta_proj = theta - self.settings["projector_offset"]["theta"]
        phi_proj = phi - self.settings["projector_offset"]["phi"]
        theta_steps = degrees_to_steps(theta_proj, self.settings["steps_per_revolution"])
        phi_steps = degrees_to_steps(phi_proj, self.settings["steps_per_revolution"])
        
        try: 
            self.moveAbsolute(theta_steps, phi_steps)
        except Exception as e:
            print(f"Error pointing projector: {e}")
            return e

    def moveAbsolute(self, theta_steps, phi_steps):
        theta_relative = theta_steps - self.current_position[0]
        phi_relative = phi_steps - self.current_position[1] #FIXME: This might be wrong
        self.current_signs = [np.sign(theta_relative), np.sign(phi_relative)]
        
        
        # theta_compensated = 0
        # phi_compensated = 0
        # if theta_relative != 0:
        #     if self.current_signs[0] != self.previous_signs[0]:
        #         #Add steps (depending on direction) to compensate for backlash
        #         theta_compensated += self.settings["backlash"]["theta"] * self.current_signs[0]
        # if phi_relative != 0:
        #     if self.current_signs[1] != self.previous_signs[1]:
        #         #Add steps (depending on direction) to compensate for backlash
        #         phi_compensated += self.settings["backlash"]["phi"] * self.current_signs[1]
        # self.previous_signs = self.current_signs
        # command = CommandGenerator.generate_point_command(theta_compensated, -phi_compensated)
        command = CommandGenerator.generate_point_command(theta_relative, -phi_relative)
        self.ser.write(command.encode())
        self.ser.flush()
        
        #Wait for a response
        while self.ser.in_waiting == 0:
            pass
        response = self.ser.readline().decode('utf-8').rstrip()
        #Check if positionining was successful
        if response != "S":
            return Exception(f"Positioning Error: {response}")
        self.current_position = [theta_steps, phi_steps]
    
    def center(self):
        self.ser.write(CommandGenerator.CENTER_COMMAND.encode())
    def laser_on(self):
        self.ser.write(CommandGenerator.LASER_ON_COMMAND.encode())
    def laser_off(self):
        self.ser.write(CommandGenerator.LASER_OFF_COMMAND.encode())
    def motors_off(self):
        self.ser.write(CommandGenerator.MOTORS_OFF_COMMAND.encode())

    #projects a cone. alpha and beta are in degrees
    def project_cone(self, alpha, beta, freq=15, duration = 3):
        """
        Sends command to the controller to project a cone with the given alpha and beta angles
        alpha and beta should be less than the range of motion of the projector
        Command sent: L 
        """
        
        alpha_int = int(alpha*255/(self.settings["projector_ROM"]["alpha"]))
        beta_int = int(beta*255/(self.settings["projector_ROM"]["beta"]))

        if alpha_int > 255 or beta_int > 255:
            return Exception(f"Alpha and Beta must be less than {self.settings['projector_ROM']['alpha']} and {self.settings['projector_ROM']['beta']} respectively")

        amp = max(alpha_int, beta_int)
        command = CommandGenerator.generate_cone_command(alpha_int, freq, duration)
        self.ser.write(command.encode())
        self.ser.flush()

        #Wait for a response
        while self.ser.in_waiting == 0:
            pass
        response = self.ser.readline().decode('utf-8').rstrip()
        #Check if positionining was successful
        print(response)
        if response != "S":
            return Exception(f"Projection Error")
    
    def zero(self):
        command = CommandGenerator.ZERO_COMMAND
        self.ser.write(command.encode())
        self.ser.flush()

        #Wait for a response
        while self.ser.in_waiting == 0:
            pass
        response = self.ser.readline().decode('utf-8').rstrip()
        #Check if positionining was successful
        if response != "S":
            return Exception(f"Zeroing Error")
        self.current_position = [0, 0]

    def home(self):
        self.moveAbsolute(0, 0) #Jank

class ControllerStandIn(Controller):
    def __init__(self, controller_settings):
        self.settings = controller_settings
        self.is_open = True
        self.current_position = [0, 0]

    def moveAbsolute(self, theta_steps, phi_steps):
        theta_relative = theta_steps - self.current_position[0]
        phi_relative = phi_steps - self.current_position[1] #FIXME: This might be wrong
        command = CommandGenerator.generate_point_command(theta_relative, phi_relative)
        self.current_position = [theta_steps, phi_steps]

class CommandGenerator():
    @staticmethod
    def generate_point_command(theta_steps, phi_steps):
        return f"P {theta_steps} {phi_steps}\n"
        
    def generate_cone_command(mag_int, freq_int, dur_int):
        return f"L {mag_int} {freq_int} {dur_int}\n"
    
    HOME_COMMAND = "H2\n"
    ZERO_COMMAND = "Z2\n"
    CENTER_COMMAND = "A\n"
    LASER_ON_COMMAND = "X\n"
    LASER_OFF_COMMAND = "C\n"
    MOTORS_OFF_COMMAND = "Z\n"

def degrees_to_steps(degrees, steps_per_revolution):
    return np.round(degrees * steps_per_revolution / 360)

def steps_to_degrees(steps, steps_per_revolution):
    return steps * 360 / steps_per_revolution

if os.getenv("TESTING", False) == "True":
    import yaml

    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    c = Controller(config["controller_settings"])
    if not c.is_open:
        print("Controller is not connected")
        exit()

    #make array of points, theta 0 to 90, phi 0 to 90, in 10 degree increments
     
