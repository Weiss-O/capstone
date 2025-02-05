/*
  Example sketch to control a 28BYJ-48 stepper motor
  with ULN2003 driver board and Arduino UNO using the AccelStepper library.
  More info: https://www.makerguides.com
*/

#include <AccelStepper.h>

#define motorInterfaceType 4
const int stepsPerRevolution = 2048; // Steps for full rotation
const float NUM_DEGREES = 22.5;      // Degrees to move
int STEPS = int(NUM_DEGREES * stepsPerRevolution / 360);

// Create two AccelStepper objects for tilt and pan motors
AccelStepper tilt(motorInterfaceType, 8, 10, 9, 11); // Tilt motor
AccelStepper pan(motorInterfaceType, 4, 6, 5, 7);    // Pan motor

int motorSelector = 0; // 0 for pan, 1 for tilt

void setup() {
  Serial.begin(9600);

  // Set max speed and acceleration for both motors
  tilt.setMaxSpeed(1000);
  tilt.setAcceleration(100);

  pan.setMaxSpeed(1000);
  pan.setAcceleration(100);
}

void loop() {
  // Select the motor based on motorSelector
  AccelStepper* currentMotor = (motorSelector == 0) ? &pan : &tilt;

  // Step NUM_DEGREES clockwise
  Serial.print("Motor ");
  Serial.print(motorSelector == 0 ? "Pan" : "Tilt");
  Serial.println(" - Clockwise");
  
  
  currentMotor->moveTo(STEPS);
  while (currentMotor->distanceToGo() != 0) {
    currentMotor->run();
  }
  delay(500);

  // Step NUM_DEGREES counterclockwise
  Serial.println("Counterclockwise");
  currentMotor->moveTo(-STEPS);
  while (currentMotor->distanceToGo() != 0) {
    currentMotor->run();
  }
  delay(500);

  // Return to original position
  Serial.println("Clockwise");
  currentMotor->moveTo(0);
  while (currentMotor->distanceToGo() != 0) {
    currentMotor->run();
  }

  // Toggle motor selector
  motorSelector = 1 - motorSelector; // Toggles between 0 and 1
}
