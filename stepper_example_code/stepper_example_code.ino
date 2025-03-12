
/*
 Stepper Motor Control - one revolution

 This program drives a unipolar or bipolar stepper motor.
 The motor is attached to digital pins 8 - 11 of the Arduino.

 The motor should revolve one revolution in one direction, then
 one revolution in the other direction.


 Created 11 Mar. 2007
 Modified 30 Nov. 2009
 by Tom Igoe

 */

#include <Stepper.h>
#include "config.h"
const int stepsPerRevolution = 1024;  // change this to fit the number of steps per revolution
// for your motor

// initialize the stepper library on pins 8 through 11:
Stepper myStepper(stepsPerRevolution, STEPPER_PAN_1, STEPPER_PAN_2, STEPPER_PAN_3, STEPPER_PAN_4);

void setup() {
  // set the speed at 60 rpm:
  myStepper.setSpeed(20);
  // initialize the serial port:
  Serial.begin(9600);
  myStepper.step(stepsPerRevolution/18); //move 20 degrees
  delay(500);
}

void loop() {
  // step one revolution  in one direction:
}

