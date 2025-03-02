#include "config.h"
#include "communication.h"
#include "project.h"
#include "stepper_point.h"

void setup() {
  // put your setup code here, to run once:
  init_project();
  init_stepper();
  init_comms();

  // setup the fan pin
  pinMode(FAN_PIN, OUTPUT);

  //turn on the fan here
  digitalWrite(FAN_PIN, HIGH);

}

void loop() {
  // repeatedly check for something to read from the Pi
  while (!Serial.available()) {}
  String command = Serial.readStringUntil('\n'); // Read command
  pi_communications(command);
}
