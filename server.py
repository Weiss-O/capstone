import socket
import threading

class PredictorServer:
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

if __name__ == "__main__":
    server = PredictorServer()
    server.start()
