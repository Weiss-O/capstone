import yaml
with open('config.yaml') as file:
    config = yaml.safe_load(file)


#Import the necessary modules
import os
import time
import datetime

import Camera
import Controller
import Server

if os.environ.get('RPI', 'False').lower()== 'true':
    import socket
    HOST = config["server_settings"]["HOST"]
    PORT = config["server_settings"]["PORT"]
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.connect((HOST, PORT))


#Initialize the camera
    cameraType = Camera.PiCamera if os.environ.get('RPI', 'False').lower() == 'true' else Camera.CameraStandIn #TODO: Make it so that this takes photo with  
    camera = cameraType(config["camera_settings"])
    camera.start()
    
    #Initialize the controller
    controllerType = Controller.Controller if os.environ.get('RPI', 'False').lower() == 'true' else Controller.ControllerStandIn
    teensy = controllerType(config["controller_settings"]) #TODO: Implement this class
    if not teensy.is_open:
        raise Exception("Controller not connected")
    time.sleep(2)


test_date_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
Server.send_bytes(server, b'NEW_DIR')
Server.send_bytes(server, test_date_string.encode())

while True:
    take_scan = input("Take scan? (Y/n)")
    if take_scan == "n":
        teensy.home()
        break

    for POSID in config["baseline"].keys():
        try:
            pos = config["baseline"][POSID]["camera_pos"] #Get pos coords
            theta_actual, phi_actual = teensy.point_camera(pos[0], pos[1]) #Move to pos
            time.sleep(1) #Let camera settle
            camera.capture_and_send_remote(server, "", pos=pos)
        except:
            teensy.home()
            exit()




