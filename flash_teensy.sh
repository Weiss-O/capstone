#!/bin/bash

PROJECT_DIR=/home/weiso6959/capstone/BrightSpot_v1
BRANCH=aaron
REMOTE=origin
SUBDIR=BrightSpot_v1
FLASK_SCRIPT=/home/weiso6959/capstone/python/teensy_server.py  # Change this to your actual script path
VENV_PATH=/home/weiso6959/flask_server/bin/activate  # Path to virtual environment
FLASK_LOG=/home/weiso6959/flask_server/flask.log  # Log output file

cd $PROJECT_DIR

# Function to stop the Flask server
stop_flask() {
    echo "Stopping Flask server..."
    pkill -f "$FLASK_SCRIPT"  # Kills any process running the Flask script
}

# Function to start the Flask server
start_flask() {
    echo "Starting Flask server..."
    source $VENV_PATH  # Activate the virtual environment
    nohup python3 $FLASK_SCRIPT > $FLASK_LOG 2>&1 &  # Run Flask in the background
    disown  # Prevents the process from being killed when the terminal closes
}

start_flask
# Run in an infinite loop
while true; do
    git fetch $REMOTE $BRANCH

    # Check if there are new commits affecting only the BrightSpot_v1 directory
    if ! git diff HEAD $REMOTE/$BRANCH -- $SUBDIR; then
        echo "New update detected in $SUBDIR, pulling changes..."
        git pull $REMOTE $BRANCH

        # Stop Flask before flashing
        stop_flask

        # Compile and flash
        echo "Compiling and Flashing..."
        arduino-cli compile --fqbn teensy:avr:teensy40 --output-dir build $SUBDIR
        sudo teensy_loader_cli --mcu=TEENSY40 -w -v build/*.hex -s
        echo "Flashing complete!"

        # Restart Flask
        start_flask
    fi

    sleep 10  # Wait 10 seconds before checking again
done
