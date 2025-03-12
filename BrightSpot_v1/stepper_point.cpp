#include "stepper_point.h"
#include "config.h"
#include <Arduino.h>
#include <AccelStepper.h>

#define TILT_STEPPER_SPEED 150
#define TILT_STEPPER_ACCEL 2000
#define PAN_STEPPER_SPEED 150
#define PAN_STEPPER_ACCEL 2000

const int stepsPerRevolution = 2048; // Steps for full rotation
const int positionLimit = -55; // corresponds to 10 degrees past the limit switch (10/360)*2048

// Create two AccelStepper objects for tilt and pan motors
AccelStepper tilt(STEPPER_TYPE, STEPPER_TILT_1, STEPPER_TILT_3, STEPPER_TILT_2, STEPPER_TILT_4); // Tilt motor
AccelStepper pan(STEPPER_TYPE, STEPPER_PAN_1, STEPPER_PAN_3, STEPPER_PAN_2, STEPPER_PAN_4);    // Pan motor

void init_stepper() {
// Set max speed and acceleration for both motors
  tilt.setMaxSpeed(TILT_STEPPER_SPEED);
  tilt.setAcceleration(TILT_STEPPER_ACCEL);

  pan.setMaxSpeed(PAN_STEPPER_SPEED);
  pan.setAcceleration(PAN_STEPPER_ACCEL);

  pinMode(SWITCH_TILT, INPUT);
  pinMode(SWITCH_PAN, INPUT);
}

bool home_stepper() {
  // switches closing causes the voltage to rise from gnd to 3.3v
  pan.setSpeed(-PAN_STEPPER_SPEED);
  Serial.println("Starting to wait");
  //while(digitalRead(SWITCH_PAN) == LOW && pan.currentPosition() > positionLimit) {
  while(digitalRead(SWITCH_PAN) == LOW){
    pan.runSpeed();
  }

  Serial.println("Switch hit");

  pan.setSpeed(0);
  pan.setSpeed(PAN_STEPPER_SPEED);
  pan.disableOutputs();

  if (pan.currentPosition()  <= positionLimit) {
    return false;
  }

  else {
    return true;
  }
}

bool point_steppers(int tilt_steps, int pan_steps) {

  pan.move(pan_steps); 
  while (pan.distanceToGo() != 0) {  
      pan.run();  // Keeps moving until it reaches the target
  }
  
  tilt.move(tilt_steps); 
  while (tilt.distanceToGo() != 0) {  
      tilt.run();  // Keeps moving until it reaches the target
  }

  pan.disableOutputs();
  tilt.disableOutputs(); //GETS REALLY HOT IF YOU DO THIS BE CAREFUL!
  // fail if it doesn't hit the desired position

  return true;
}

void disableMotors() {
  pan.disableOutputs();
  tilt.disableOutputs();
  return;
}

void setSpeeds(int panSpeed, int tiltSpeed){
  pan.setMaxSpeed(panSpeed);
  tilt.setMaxSpeed(tiltSpeed);
}
void setAccels(int panAccel, int tiltAccel){
  pan.setAcceleration(panAccel);
  tilt.setAcceleration(tiltAccel);
}