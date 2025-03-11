#!/bin/bash

cd /home/weiso6959/capstone
git checkout aaron
git pull
cd BrightSpot_v1
arduino-cli compile --fqbn teensy:avr:teensy40 --output-dir build
sudo teensy_loader_cli --mcu=TEENSY40 -w -v build/*.hex -s
cd /home/weiso6959/capstone
git reset --hard
git checkout development
source /home/cs/bin/activate
export RPI=True
python
