import serial
import time

# Configure the UART communication
serial_port = serial.Serial(
    port='/dev/serial0',  # Default UART port on the Raspberry Pi
    baudrate=9600,
    timeout=1  # Timeout for reading data
)

print("Raspberry Pi ready to receive messages...")

while True:
    # Check for incoming data
    if serial_port.in_waiting > 0:
        message = serial_port.readline().decode('utf-8').strip()  # Read and decode the message
        print(f"Received message from Arduino: {message}")

        # Respond to the message
        response = "Hello from Raspberry Pi!"
        serial_port.write(response.encode('utf-8'))  # Send the response
        print(f"Sent response: {response}")

    time.sleep(0.1)  # Small delay to avoid busy-waiting
