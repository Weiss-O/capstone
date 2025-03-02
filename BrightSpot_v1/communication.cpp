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

      if (point_steppers(tiltSteps, panSteps)) {
        Serial.println("S");
      }
      else {
        Serial.println('F');
      }

    }
    else{
      Serial.println('F');
    }
  }

  else if (commandChar == 'L'){ //L dur mag freq
    // call the projection command function
    int space1 = command.indexOf(' ');          // First space
    int space2 = command.indexOf(' ', space1 + 1); // Second space
    int space3 = command.indexOf(' ', space2 + 1); // Third space

    if (space1 != -1 && space2 != -1 && space3 != -1) {
      // Extract step values for pan and tilt
      int duration = command.substring(space1 + 1, space2).toInt();
      int magnitude = command.substring(space2 + 1, space3).toInt();
      int frequency = command.substring(space3 + 1).toInt();

      if (project_circle(duration, magnitude, frequency)) {
        Serial.println("S");
      }
      else {
        Serial.println('F');
      }

    }
    else{
      Serial.println('F');
    }
  }

  else if (commandChar == 'H') {
    // call the homing function
    if(home_stepper()) {
      Serial.println("S");
    }
    else {
      Serial.println("F");
    }
  }

  else {
    //send back F to indicate that we failed to execute a command 
    Serial.println('F');
  }

  return;
}
