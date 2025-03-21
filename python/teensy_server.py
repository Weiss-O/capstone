from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import serial
import threading
import glob
import time
import os
import signal

app = Flask(__name__)
CORS(app)

messages = []  # Store received messages

FIND_DEV_TIMEOUT = 15
CONNECT_PORT_TIMEOUT = 30

settings = {
    "baudrate": 9600,
    "timeout": 1
}

def find_teensy():
    """Finds the connected Teensy device."""
    ports = glob.glob('/dev/ttyACM*')
    if ports:
        messages.append(f"Found Teensy device: {ports[0]}")
        return ports[0]
    return None

class SerialManager:
    def __init__(self):
        self.ser = None
        start_time = time.time()
        self.port = None
        messages.append("Searching for Teensy device...")
        while not self.port and time.time() - start_time < FIND_DEV_TIMEOUT:
            self.port = find_teensy()
        self.is_open = False

    def connect(self):
        """Attempts to open the serial port, if available."""
        start_time = time.time()
        messages.append("Connecting to serial port...")
        while self.port and not self.is_open and time.time() - start_time < CONNECT_PORT_TIMEOUT:
            try:
                self.ser = serial.Serial(self.port, settings["baudrate"], timeout=settings["timeout"])
                self.is_open = self.ser.is_open
                messages.append(f"Connected to {self.port}")
            except Exception as e:
                self.is_open = False
        if not self.is_open:
            messages.append("Failed to connect to serial port.")
            exit()

    def disconnect(self):
        """Closes the serial connection."""
        if self.ser:
            self.ser.close()
            self.is_open = False

    def send_command(self, command):
        """Sends a command if the serial port is open."""
        if self.is_open and self.ser:
            try:
                self.ser.write((command + "\n").encode())
                self.ser.flush()
                time.sleep(0.1)
            except Exception as e:
                messages.append(f"Serial write error: {e}")
                self.is_open = False
    
    def reconnect(self):
        """Attempts to reconnect to the serial port."""
        self.port = None
        messages.append("Lost Connection, searching for Teensy device...")
        start_time = time.time()
        while not self.port and time.time() - start_time < FIND_DEV_TIMEOUT:
            self.port = find_teensy()

        self.connect()

serial_manager = SerialManager()

def read_serial():
    """Continuously reads data from the serial port."""
    while True:
        if serial_manager.is_open and serial_manager.ser.in_waiting > 0:
            line = serial_manager.ser.readline().decode().strip()
            messages.append(line)
            if len(messages) > 300:  # Limit stored messages
                messages.pop(0)
        time.sleep(0.1)  # Prevent high CPU usage

# Start background thread to read from serial
threading.Thread(target=read_serial, daemon=True).start()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        command = request.form["command"]
        messages.append(f"Sent command: {command}")
        serial_manager.send_command(command)
        return "", 204
    return '''
        <form method="post">
            <input type="text" name="command" placeholder="Enter command">
            <button type="button" onclick="sendCommand()">Send</button>
        </form>
        <div id="logs"></div>
        <script>
            async function sendCommand() {
                let commandInput = document.querySelector("input[name='command']");
                let command = commandInput.value;
                if (!command.trim()) return;
                await fetch("/", {
                    method: "POST",
                    headers: { "Content-Type": "application/x-www-form-urlencoded" },
                    body: "command=" + encodeURIComponent(command)
                });
                commandInput.value = "";
                fetchLogs();
            }
            async function fetchLogs() {
                let response = await fetch("/logs");
                let data = await response.json();
                document.getElementById("logs").innerHTML = data.logs.join("<br>");
            }
            setInterval(fetchLogs, 1000);
        </script>
    '''

@app.route("/logs")
def get_logs():
    return jsonify({"logs": messages})

@app.route("/shutdown", methods=["POST"])
def shutdown():
    """Gracefully shuts down the server."""
    messages.append("Shutting down server...")
    serial_manager.disconnect()
    os.kill(os.getpid(), signal.SIGTERM)
    return "Shutting down...", 200

if __name__ == "__main__":
    serial_manager.connect()  # Connect on startup
    app.run(host="0.0.0.0", port=5000)
