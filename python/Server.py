import socket

def send_bytes(server:socket.socket, data):
    # If data is a string, convert to bytes
    data_length = len(data).to_bytes(4, 'big')
            
    # Send everything with lengths
    server.sendall(data_length + data)

def get_response(server:socket.socket):
    resp_len = int.from_bytes(server.recv(4), 'big') 
    response = recvall(server, resp_len)
    return response

def recvall(server:socket.socket, n):
    """Helper function to receive exactly n bytes."""
    data = b''
    while len(data) < n:
        packet = server.recv(n - len(data))
        if not packet:
            #Connection closed unexpectedly
            return None
        data += packet
    return data