#include "stepper_point.h"
#include "config.h"
#include "project.h"
#include "communication.h"
#include <Arduino.h>

void init_comms(){
  Serial.begin(SERIAL_BAUD_RATE);
}

// have a function to handle each of the incoming pi functions
void pi_communications(String command) {
  // Process the command
  char commandChar = command[0];

  if (commandChar == 'P') {
    int space1 = command.indexOf(' ');          // First space
    int space2 = command.indexOf(' ', space1 + 1); // Second space

    if (space1 != -1 && space2 != -1) {
      // Extract step values for pan and tilt
      int panSteps = command.substring(space1 + 1, space2).toInt();
      int tiltSteps = command.substring(space2 + 1).toInt();

      // Serial.print("echo stepper, ");
      // Serial.print(panSteps);
      // Serial.print(", ");
      // Serial.println(tiltSteps);

      if (point_steppers(tiltSteps, panSteps)) {
        Serial.println("S");
      }
      else {
        Serial.println('F3');
      }

    }
    else{
      Serial.println('F4');
    }
  }

  else if (commandChar == 'Z'){
    disableMotors(); //Adding comment to introduce a change for testing autoflash 8
    Serial.println("S");
  }

  else if (commandChar == 'L'){ //L dur mag freq
    // call the projection command function
    int space1 = command.indexOf(' ');          // First space
    int space2 = command.indexOf(' ', space1 + 1); // Second space
    int space3 = command.indexOf(' ', space2 + 1); // Third space

    if (space1 != -1 && space2 != -1 && space3 != -1) {
      // Extract step values for pan and tilt
      int duration = command.substring(space1 + 1, space2).toInt();
      float magnitude = command.substring(space2 + 1, space3).toInt();
      magnitude = map(magnitude, 0.0, 255.0, 0.0, 10.0);
      int frequency = command.substring(space3 + 1).toInt();

      Serial.print("echo project, ");
      Serial.print(duration);
      Serial.print(", ");
      Serial.print(magnitude);
      Serial.print(", ");
      Serial.println(frequency);

      if (project_circle(duration, magnitude, frequency)) {
        Serial.println("S");
      }
      else {
        Serial.println("Fuck");
      }

    }
    else{
      Serial.println("Fa");
    }
  }

  else if (commandChar == 'H') {
    // call the homing function
    Serial.println("echo homing");
    if(home_stepper()) {
      Serial.println("S");
    }
    else {
      Serial.println("Fb");
    }
  }

  else if (commandChar == 'A') {
    Serial.println("echo autocalibrate");
    calibrate_galvo();
  }

  else if (commandChar == 'X') {
    Serial.println("echo laser on");
    laser_on();
    Serial.println("S");
  }

  else if (commandChar == 'C') {
    Serial.println("echo laser off");
    laser_off();
    Serial.println("S");
  }
  else if(commandChar == 'S'){
    int space1 = command.indexOf(' ');
    int space2 = command.indexOf(' ', space1 + 1);
    if (space1 != -1 && space2 != -1){
      int panSpeed = command.substring(space1 + 1, space2).toInt();
      int tiltSpeed = command.substring(space2 + 1).toInt();
      setSpeeds(panSpeed, tiltSpeed);
    }
  }
  else if(commandChar == 'Y'){
    int space1 = command.indexOf(' ');
    int space2 = command.indexOf(' ', space1 + 1);
    if (space1 != -1 && space2 != -1){
      int panAccel = command.substring(space1 + 1, space2).toInt();
      int tiltAccel = command.substring(space2 + 1).toInt();
      setAccels(panAccel, tiltAccel);
    }
  }
  else {
    //send back F to indicate that we failed to execute a command 
    Serial.println('F1');
  }

  return;
}
