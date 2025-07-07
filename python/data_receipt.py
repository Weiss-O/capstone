import cv2
import numpy as np
import socket
import yaml

import Server

import os

with open('config.yaml') as file:
    config = yaml.safe_load(file)

root_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(root_dir, 'scans')

count = 0
def handle_client(s):
    try:
        while True:
            command = Server.get_response(s)
            if command is None:
                print("Connection closed unexpectedly")
                break
            print(f"Received command: {command}")
            if command == b'NEW_DIR':
                dir_name = Server.get_response(s)
                print(f"Creating new directory: {dir_name}")
                
                dir_name = os.path.join(root_dir, dir_name.decode())
                os.makedirs(dir_name)
                count = 0
            elif command == b'STORE_IMAGE':
                #Receive the image
                image_bytes = Server.get_response(s)
                image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
                print(f"Received image of size {len(image_bytes)} bytes")

                #Wait for the client to send the image name
                camera_pos = Server.recv_coords(s) #arry with 2 floats
                #Create directory named after position
                pos = f"{camera_pos[0]:.2f}_{camera_pos[1]:.2f}"
                os.makedirs(os.path.join(dir_name, pos), exist_ok=True)
                print(f"Received scan at pos: {pos}")

                #round up count to nearest multiple of 4
                scan_count = int(count/4)
                #Save the image
                img_name = f"img_{scan_count}.jpg"
                img_path = os.path.join(dir_name, pos, img_name)
                cv2.imwrite(img_path, image)
                print(f"Saved image to {img_path}")
                Server.send_bytes(client_socket, b'ACK')
                count += 1

            else:
                print(f"Unknown command: {command}")
                break

    except Exception as e:
        print(f"Error handling client: {e}")

if __name__ == "__main__":
    host = config['server_settings']['HOST']
    port = config['server_settings']['PORT']

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen()

    print(f"Server listening on {host}:{port}")

    while True:
        client_socket, addr = server_socket.accept()
        print(f"Connection from {addr}")
        handle_client(client_socket)