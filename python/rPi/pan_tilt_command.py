#!/usr/bin/env python3
import serial # Import the serial library
import time

if __name__ == '__main__':
    ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
    ser.reset_input_buffer()

    while True:
        #Wait for me to type a command into the command line and then send that command to the Arduino
        command = input("Type positioning command->") + "\n"
        ser.write(command.encode())
        #Wait for the Arduino to respond with timeout
        start_time = time.time()
        timeout = 10  # 5 second timeout
        while ser.in_waiting == 0:
            if time.time() - start_time > timeout:
                print("Error: Timeout waiting for Arduino response")
                break
        #Read the response from the Arduino
        line = ser.readline().decode('utf-8').rstrip()
        print(line)

