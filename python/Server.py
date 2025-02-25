import socket
import struct

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

def send_coords(server:socket.socket, data):
    """For sending [theta, phi] camera coordinates to the server"""
    packed_data = struct.pack('>2f', *data)
    send_bytes(server, packed_data)

def recv_coords(server:socket.socket):
    """For receiving [theta, phi] camera coordinates"""
    data = get_response(server)
    return struct.unpack('>2f', data)

def send_detections(server:socket.socket, detection_data):
    """Send detection bounding boxes to the client"""
    num_detections = len(detection_data) #Number of detections
    packed_data = num_detections.to_bytes(4, 'big')
    for data in detection_data:
        packed_data += struct.pack('>4i', *data)

    send_bytes(server, packed_data)

def recv_detections(server:socket.socket):
    """Receive detection bounding boxes from the server"""
    raw_data = get_response(server)
    num_detections = int.from_bytes(raw_data[:4], 'big')

    offset = 4
    detection_data = []
    for _ in range(num_detections):
        #Read 4 integers from the raw data
        x, y, w, h = struct.unpack('>4i', raw_data[offset:offset+16])
        detection_data.append([x, y, w, h])
        offset += 16

    return detection_data