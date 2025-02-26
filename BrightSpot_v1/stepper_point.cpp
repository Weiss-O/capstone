#include "stepper_point.h"
#include "config.h"
#include <Arduino.h>
#include <AccelStepper.h>

#define STEPPER_SPEED 1000
#define STEPPER_ACCEL 100

const int stepsPerRevolution = 2048; // Steps for full rotation
const int positionLimit = -55 // corresponds to 10 degrees past the limit switch (10/360)*2048

// Create two AccelStepper objects for tilt and pan motors
AccelStepper tilt(STEPPER_TYPE, STEPPER_TILT_1, STEPPER_TILT_2, STEPPER_TILT_3, STEPPER_TILT_4); // Tilt motor
AccelStepper pan(STEPPER_TYPE, STEPPER_PAN_1, STEPPER_PAN_2, STEPPER_PAN_3, STEPPER_PAN_4);    // Pan motor

void init_stepper() {
// Set max speed and acceleration for both motors
  tilt.setMaxSpeed(STEPPER_SPEED);
  tilt.setAcceleration(STEPPER_ACCEL);

  pan.setMaxSpeed(STEPPER_SPEED);
  pan.setAcceleration(STEPPER_ACCEL);

  pinMode(SWITCH_TILT, INPUT);
  pinMode(SWTICH_PAN, INPUT)
}

bool home_stepper() {
  // switches closing causes the voltage to rise from gnd to 3.3v
  pan.setSpeed(-STEPPER_SPEED)
  while(digitalRead(SWITCH_PAN) == LOW && pan.currentPosition() > positionLimit) {
    pan.runSpeed();
  }
  
  pan.setSpeed(0);
  pan.setSpeed(STEPPER_SPEED)
  
  if (pan.currentPosition()  <= positionLimit) {
    return false
  }

  else {
    return true;
  }
}

bool point_steppers(int tilt_deg, int pan_deg) {

  pan.move(pan_stpes); 
  while (pan.isRunning()) {  
      pan.run();  // Keeps moving until it reaches the target
  }
  
  tilt.move(tilt_steps); 
  while (tilt.isRunning()) {  
      tilt.run();  // Keeps moving until it reaches the target
  }

  return true;
}