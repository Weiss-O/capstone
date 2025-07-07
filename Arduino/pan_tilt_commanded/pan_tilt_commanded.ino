#include <AccelStepper.h>

#define motorInterfaceType 4
const int stepsPerRevolution = 2048; // Steps for full rotation
const float NUM_DEGREES = 22.5;      // Degrees to move
int STEPS = int(NUM_DEGREES * stepsPerRevolution / 360);

// Create two AccelStepper objects for tilt and pan motors
AccelStepper tilt(motorInterfaceType, 8, 10, 9, 11); // Tilt motor
AccelStepper pan(motorInterfaceType, 4, 6, 5, 7);    // Pan motor

void setup() {
  Serial.begin(9600);

  // Set max speed and acceleration for both motors
  tilt.setMaxSpeed(1000);
  tilt.setAcceleration(100);

  pan.setMaxSpeed(1000);
  pan.setAcceleration(100);
}

void loop() {
  // Wait for a command from the Pi
  while (!Serial.available()) {}
  String command = Serial.readStringUntil('\n'); // Read command

  // Process the command
  if (command[0] == 'P') {
    // 'P' (Position) command: Move motors by relative steps
    int space1 = command.indexOf(' ');          // First space
    int space2 = command.indexOf(' ', space1 + 1); // Second space

    if (space1 != -1 && space2 != -1) {
      // Extract step values for pan and tilt
      int panSteps = command.substring(space1 + 1, space2).toInt();
      int tiltSteps = command.substring(space2 + 1).toInt();

      // Move pan motor
      pan.move(panSteps);
      while (pan.distanceToGo() != 0) {
        pan.run();
      }

      // Move tilt motor
      tilt.move(tiltSteps);
      while (tilt.distanceToGo() != 0) {
        tilt.run();
      }

      Serial.println("SUCCESS");
    }
  } else if (command[0] == 'Z') {
    // 'Z' (Zero) command: Set current position to 0
    int motorCode = command.substring(1).toInt();
    if (motorCode == 0) {
      pan.setCurrentPosition(0);
    } else if (motorCode == 1) {
      tilt.setCurrentPosition(0);
    } else if (motorCode == 2) {
      pan.setCurrentPosition(0);
      tilt.setCurrentPosition(0);
    }
    Serial.println("SUCCESS");
  } else if (command[0] == 'H') {
    // 'H' (Home) command: Move motors to home (position 0)
    int motorCode = command.substring(1).toInt();
    if (motorCode == 0) {
      pan.moveTo(0);
      while (pan.distanceToGo() != 0) {
        pan.run();
      }
    } else if (motorCode == 1) {
      tilt.moveTo(0);
      while (tilt.distanceToGo() != 0) {
        tilt.run();
      }
    } else if (motorCode == 2) {
      pan.moveTo(0);
      while (pan.distanceToGo() != 0) {
        pan.run();
      }
      tilt.moveTo(0);
      while (tilt.distanceToGo() != 0) {
        tilt.run();
      }
    }
    Serial.println("SUCCESS");
  } else if (command[0] == 'S') {
    // 'S' (Stop) command: Stop both motors
    pan.stop();
    tilt.stop();
    Serial.println("SUCCESS");
  } else if (command[0] == 'L'){
    int space1 = command.indexOf(' ');          // First space
    int space2 = command.indexOf(' ', space1 + 1); // Second space
    int space3 = command.indexOf(' ', space2 + 1); // Third space

    Serial.println("SUCCESS");
  }
}
