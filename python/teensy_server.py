from flask import Flask, request, render_template, jsonify
import serial
import threading

app = Flask(__name__)

# Change this to match your Teensy's serial port
SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE = 115200

ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
messages = []  # Store received messages

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