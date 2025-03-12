import time
import glob
import serial

if __name__ == '__main__':
    # Search for available ttyACM devices
    ports = glob.glob('/dev/ttyACM*')
    if not ports:
        raise IOError("No ttyACM devices found")
    
    ser = serial.Serial(ports[0], 9600, timeout=1)
    ser.reset_input_buffer()

    # Get user command and send it to Arduino
    command = input("Type positioning command->") + "\n"
    ser.write(command.encode())
    
    # Wait for and print responses continuously
    print("Listening for responses...")
    try:
        while True:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').rstrip()
                print(line)
    except KeyboardInterrupt:
        print("Stopped listening.")
    finally:
        ser.close()
