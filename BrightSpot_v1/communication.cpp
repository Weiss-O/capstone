#include "stepper_point.h"
#include "config.h"
#include "project.h"
#include <Arduino.h>

// have a function to handle each of the incoming pi functions
void pi_communications(string command){
  // Process the command
  commandChar = command[0]

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

  else if (commandChar == 'L'){
    // call the projection command function
  }

  else if (commandChar == 'H') {
    // call the homing function
  }

  else {
    //send back F to indicate that we failed to execute a command 
    Serial.println('F');
  }

  return;
}

// projection - receive info and call the function