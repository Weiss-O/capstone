#!/bin/bash

# Ensure an argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <arduino_project_name>"
    exit 1
fi

# Assign the first argument to a variable
arduino_project_name=$1

cd /home/weiso6959/capstone

# Ensure clean checkout
git checkout -f aaron
git pull

# Navigate to the specified Arduino project directory
cd "$arduino_project_name" || { echo "Project directory not found!"; exit 1; }

# Ensure build directory exists
mkdir -p build

# Compile using arduino-cli
arduino-cli compile --fqbn teensy:avr:teensy40 --output-dir build

# Flash firmware to Teensy
sudo teensy_loader_cli --mcu=TEENSY40 -w -v build/*.hex -s

# Reset repo (only for capstone directory)
cd /home/weiso6959/capstone
git reset --hard

# Switch back to development branch
git checkout development

# Activate Python virtual environment
if [ -f "/home/cs/bin/activate" ]; then
    source /home/cs/bin/activate
else
    echo "Virtual environment not found!"
fi

# Set environment variable and start Python
export RPI=True
python
