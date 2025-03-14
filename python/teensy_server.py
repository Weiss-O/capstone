from flask import Flask, request, render_template, jsonify
import serial
import threading
import glob
import time


app = Flask(__name__)

messages = []  # Store received messages

settings = {
    "baudrate": 9600,
    "timeout": 1
}
messages.append("Waiting until port is detected")
ports = None
start_time = time.time()
timeout = 20  # Timeout in seconds

while not ports and (time.time() - start_time) < timeout:
    ports = glob.glob('/dev/ttyACM*')
    time.sleep(1)
messages.append(f"Detected teensy on port: {ports[0]}")

if not ports:
    raise Exception("No serial ports found within the timeout period")
    
port = ports[0]
is_open = False
start_time = time.time()
while not is_open and (time.time() - start_time) < timeout:
    try:
        ser = serial.Serial(port, settings["baudrate"], timeout = settings["timeout"])
        is_open = ser.is_open
    except:
        time.sleep(1)
if not is_open:
    raise Exception("Failed to open serial port")
messages.append(f"Opened serial port: {port}")
ser.reset_input_buffer()

def read_serial():
    global messages
    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode().strip()
            messages.append(line)
            if len(messages) > 100:  # Limit stored messages
                messages.pop(0)

# Start background thread to read from serial
threading.Thread(target=read_serial, daemon=True).start()

def read_serial():
    global messages
    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode().strip()
            messages.append(line)
            if len(messages) > 100:  # Limit stored messages
                messages.pop(0)
        time.sleep(0.1)  # Prevent high CPU usage

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        command = request.form["command"]
        ser.write((command + "\n").encode())
        ser.flush()
        time.sleep(0.1)
        return '', 204

    return '''
        <form method="post">
            <input type="text" name="command" placeholder="Enter command">
            <button type="button" onclick="sendCommand()">Send</button>
        </form>
        <div id="logs"></div>
        <script>
            async function sendCommand() {
                let command = document.querySelector("input[name='command']").value;
                let response = await fetch("/", {
                    method: "POST",
                    headers: { "Content-Type": "application/x-www-form-urlencoded" },
                    body: "command=" + encodeURIComponent(command)
                });
                document.querySelector("input[name='command']").value = "";
            }
            document.querySelector("button").onclick = sendCommand;
            
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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)