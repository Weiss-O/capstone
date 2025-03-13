import socket
import Classifier
import cv2
import numpy as np
import struct

predictors = {}

def recvall(sock, n):
    # Receive n bytes from a socket
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            #Connection closed unexpectedly
            return None
        data += packet
    return data

def recv_message(sock):
    # Receive a message from a socket
    length_bytes = recvall(sock, 4)
    if not length_bytes:
        return None
    message_length = int.from_bytes(length_bytes, 'big')

    # Now read the full message based on length
    message = recvall(sock, message_length)
    return message

def send_prediction_results(sock, results):
    """
    Sends list of pred resuts
    each is a [mask, score] pair

    The protocol is:
    - 4 bytes for number of results
    - For each result:
        - 4 bytes heighgt
        - 4 bytes width
        - height*width bytes: mask data
        - 4 bytes for score
        - 4 bytes: score
    """
    num_results = len(results)
    sock.sendall(num_results.to_bytes(4, 'big'))
    for mask, score in results:
        mask = mask.astype(np.uint8)
        height, width = mask.shape
        sock.sendall(height.to_bytes(4, 'big'))
        sock.sendall(width.to_bytes(4, 'big'))
        mask_bytes = mask.tobytes()
        mask_size = len(mask_bytes).to_bytes(4, 'big')
        print(f"Sending mask of size {int.from_bytes(mask_size, 'big')} bytes")
        sock.sendall(mask_size)
        sock.sendall(mask_bytes)
        sock.sendall(struct.pack('!f', score))

def handle_client(client_socket):
    try:
        while True:
            #Receive the command
            command = recv_message(client_socket)
            if command is None:
                print("Client disconnected")
                break
            print(f"Received command: {command}")

            if command == b'INIT_PREDICTOR':
                # Acknowledge the INIT_PREDICTOR command
                ack = b'INIT_PREDICTOR_ACK'
                client_socket.sendall(len(ack).to_bytes(4, 'big') + ack)

                # Receive the predictor ID
                predictor_id = recv_message(client_socket)
                print(f"Registered predictor ID: {predictor_id}")
                if predictor_id is None:
                    print("Did not receive predictor ID")
                    break
                
                predictors[predictor_id] = Classifier.SAM2Predictor()

                #Send back ID registration acknowledgment
                ack = b'ID_ACK'
                client_socket.sendall(len(ack).to_bytes(4, 'big') + ack)
            
            elif command == b'SET_IMAGE':
                # Receive ID
                predictor_id = recv_message(client_socket)
                print(f"Predictor ID: {predictor_id}")
                if predictor_id is None:
                    print("Did not receive predictor ID")
                    break
                print(f"ID for set Image command {predictor_id}")

                #Receive the image size (4 bytes)
                image_size_bytes = recvall(client_socket, 4)
                if not image_size_bytes:
                    print("Did not receive image size")
                    break

                image_size = int.from_bytes(image_size_bytes, 'big')
                print(f"Expecting image of size: {image_size} bytes")

                #receive the image data in full
                image_data = recvall(client_socket, image_size)
                if not image_data:
                    print("Did not receive image data")
                    break
                
                #process the image data TODO: Implement this
                image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
                print(f"Received image data ({len(image_data)}) bytes")

                #Set the image in the predictor
                predictors[predictor_id].set_image(image)

                #Send back acknowledgement
                ack = b'SET_IMAGE_ACK'
                client_socket.sendall(len(ack).to_bytes(4, 'big') + ack)

            elif command == b'PREDICT':
                #Receive ID
                predictor_id = recv_message(client_socket)
                if predictor_id is None:
                    print("Did not receive predictor ID")
                    break
                print(f"Predictor ID for Predict command: {predictor_id}")

                prompt_data = recv_message(client_socket)
                if prompt_data is None:
                    print("Did not receive prompt data")
                    break
                num_prompts = int.from_bytes(prompt_data[:4], 'big')
                prompts = []
                offset = 4
                for _ in range(num_prompts):
                    x = int.from_bytes(prompt_data[offset:offset+4], 'big')
                    y = int.from_bytes(prompt_data[offset+4:offset+8], 'big')
                    prompts.append((x, y))
                    offset += 8
                print(f"Received {num_prompts} prompts: {prompts}")

                #TODO: Implement prediction computation and prepare response data
                results = predictors[predictor_id].predict(prompts=prompts)
                ack = b'PREDICT_ACK'
                client_socket.sendall(len(ack).to_bytes(4, 'big') + ack)

                print("sending_results")
                #Send the results
                send_prediction_results(client_socket, results)
            else:
                print(f"Unknown command: {command}")
                break
    except Exception as e:
        print(f"Client error: {e}")
    finally:
        client_socket.close()
                
"""
    def __init__(self, host='0.0.0.0', port=5000):
        self.host = host
        self.port = port
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        print(f"Server listening on {self.host}:{self.port}")
        
    def start(self):
        while True:
            client_socket, addr = self.server_socket.accept()
            print(f"Connection from {addr}")
            threading.Thread(target=self.handle_client, args=(client_socket,)).start()
    
    def handle_client(self, client_socket):
        try:
            while True:
                command = self.receive_message(client_socket)
                if command == b'INIT_PREDICTOR':
                    print("Received INIT_PREDICTOR")
                    self.send_message(client_socket, b'INIT_PREDICTOR_ACK')
                    predictor_id = self.receive_message(client_socket)
                    print(f"Registered predictor ID: {predictor_id}")
                    self.send_message(client_socket, b'ID_ACK')
                elif command == b'SET_IMAGE':
                    print("Received SET_IMAGE")
                    predictor_id = self.receive_message(client_socket)
                    print(f"Predictor ID: {predictor_id}")
                    image_size = int.from_bytes(self.receive_message(client_socket), 'big')
                    print(f"Expecting image of size: {image_size} bytes")
                    received_bytes = 0
                    while received_bytes < image_size:
                        chunk = client_socket.recv(min(8192, image_size - received_bytes))
                        if not chunk:
                            break
                        received_bytes += len(chunk)
                    print("Image received.")
                    self.send_message(client_socket, b'SET_IMAGE_ACK')
                elif command == b'PREDICT':
                    print("Received PREDICT")
                    predictor_id = self.receive_message(client_socket)
                    print(f"Predictor ID: {predictor_id}")
                    num_prompts = int.from_bytes(self.receive_message(client_socket), 'big')
                    prompts = []
                    for _ in range(num_prompts):
                        x = int.from_bytes(self.receive_message(client_socket), 'big')
                        y = int.from_bytes(self.receive_message(client_socket), 'big')
                        prompts.append((x, y))
                    print(f"Received {num_prompts} prompts: {prompts}")
                    self.send_message(client_socket, b'PREDICT_ACK')
                else:
                    print(f"Unknown command: {command}")
                    break
        except Exception as e:
            print(f"Client error: {e}")
        finally:
            client_socket.close()
    
    def receive_message(self, client_socket):
        message_length = int.from_bytes(client_socket.recv(4), 'big')
        return client_socket.recv(message_length)
    
    def send_message(self, client_socket, message):
        client_socket.sendall(len(message).to_bytes(4, 'big') + message)
"""

if __name__ == "__main__":
    import yaml
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    
    host = config['server_settings']['HOST']
    port = config['server_settings']['PORT']

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(5)
    print(f"Server listening on {host}:{port}")

    while True:
        client_socket, addr = server_socket.accept()
        print(f"Connection from {addr}")
        handle_client(client_socket)