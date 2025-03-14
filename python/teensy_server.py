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

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        command = request.form["command"]
        ser.write((command + "\n").encode())  # Send command over serial
        return f"Sent: {command}"

    return '''
        <form method="post">
            <input type="text" name="command" placeholder="Enter command">
            <button type="submit">Send</button>
        </form>
        <div id="logs"></div>
        <script>
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