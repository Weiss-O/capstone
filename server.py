import socket
import os

HOST = '0.0.0.0'  # Listen on all interfaces
PORT = 5000
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "test.jpg")

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
    server.bind((HOST, PORT))
    server.listen()
    print(f"Server listening on {HOST}:{PORT}")
    
    conn, addr = server.accept()
    with conn:
        print(f"Connected by {addr}")
        with open(output_path, "wb") as output_file:
            while True:
                data = conn.recv(1024)  # Receive data in chunks
                if not data:
                    break
                output_file.write(data)  # Write data to the file

    print(f"Image received and saved to {output_path}")
