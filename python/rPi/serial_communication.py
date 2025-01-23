#Demo code to test communication between raspberry pi and arduino

import serial

if __name__ == "__main__":
    ser = serial.Serial('/dev/ttyACM1', 9600, timeout=1)
    ser.flush()

    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').rstrip()
            print(line)
            if float(line) > 60 and float(line)<70:
                ser.write(b"blue\n")
            elif float(line) > 70:
                ser.write(b"white\n")
            else:
                ser.write(b"all\n")

        
            