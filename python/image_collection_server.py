#Script to collect images to test image alignment methods

import cv2
import numpy as np
import socket
import yaml
import os
import Server

image_directory = "alignment_test_images"

with open('config.yaml') as file:
    config = yaml.safe_load(file)


def handle_client(client_socket):
    try:
        while True:
            #Wait for client to send image
            image_bytes = Server.get_response(client_socket)
            if image_bytes is None:
                print("Connection closed unexpectedly")
                break
            #Decode image
            image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            #Wait for the client to send the image name
            image_name = Server.get_response(client_socket)
            print(f"Received image: {image_name}")
            #Save the image
            cv2.imwrite(f"{image_directory}/{image_name.decode()}.jpg", image)

            #Send ACK
            Server.send_bytes(client_socket, b"ACK")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        client_socket.close()

if __name__ == "__main__":
    host = config['server_settings']['HOST']
    port = config['server_settings']['PORT']

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen()

    print(f"Server listening on {host}:{port}")

    #Make directory
    os.makedirs(image_directory, exist_ok=True)

    while True:
        client_socket, addr = server_socket.accept()
        print(f"Connection from {addr}")
        handle_client(client_socket)