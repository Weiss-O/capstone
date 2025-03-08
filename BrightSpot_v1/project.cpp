#include "project.h"
#include "config.h"
#include <Arduino.h>

const float kp_x = 200.0;
const float ki_x = 0.1;
const float kd_x = 100;

const float kp_y = 600.0;
const float ki_y = 0.1;
const float kd_y = 150;

const float kp_c = 100;
const float ki_c = 1;
const float kd_c = 0.0;

float sumErr_x = 0.0;
float e_prev_x = 0.0;

float sumErr_y = 0.0;
float e_prev_y = 0.0;

const float Ts = 500; // sample time in micros
const int minPWM = 2000;
const int pwmMax = 32757;
const int threshold = 200;

const float max_y = 732.0;
const float min_y = 282.0;
const float max_x = 600.0;
const float min_x = 385.0;

const float slope_y = 54.0/(max_y - min_y);
const float slope_x = 54.0/(max_x - min_x);
const float offsetx = -27 - (slope_x*min_x)+6;
const float offsety = -27 - (slope_y*min_y);

float circle_control_PID(float e_new, bool is_x) {
  if (is_x) {
    sumErr_x += e_new;
    float dErr = (e_new - e_prev_x)/Ts;
    e_prev_x = e_new;
    return kp_x * e_new + ki_x * sumErr_x + kd_x * dErr;
  }

  else {
    sumErr_y += e_new;
    float dErr = (e_new - e_prev_y)/Ts;
    e_prev_y = e_new;
    return kp_y * e_new + ki_y * sumErr_y + kd_y * dErr;
  }
}

bool center_mirrors_PID(float threshold_error) {
  // Serial.println("centering started");
  int start_time = micros();

  do {
    int loop_start = micros();

    // check if we've timed out
    if (loop_start - start_time > 20000000){
      command_motors(GALVO_MOTOR_X1, GALVO_MOTOR_X2, 0.0);
      command_motors(GALVO_MOTOR_Y1, GALVO_MOTOR_Y2, 0.0);
      return false;
    }

    MirrorAngles test = get_mirror_angles();
    float mirrorAnglex = test.anglex;
    float mirrorAngley = test.angley;

    // calculate error
    float error_x = (0.0 - mirrorAnglex);
    float error_y = (0.0 - mirrorAngley);

    sumErr_y += error_y;
    sumErr_x += error_x;

    float dErr_x = (error_x - e_prev_x)/Ts;
    float dErr_y = (error_y - e_prev_y)/Ts;

    // apply PID loop
    float command_x = kp_c * error_x + ki_c * sumErr_x + kd_c * dErr_x;
    float command_y = kp_c * error_y + ki_c * sumErr_y + kd_c * dErr_y;

    float pwm_x = command_motors(GALVO_MOTOR_X1, GALVO_MOTOR_X2, command_x);
    float pwm_y = command_motors(GALVO_MOTOR_Y1, GALVO_MOTOR_Y2, command_y);

    // shift enew to eprev
    e_prev_y = error_y;
    e_prev_x = error_x;

    // Serial.print(mirrorAnglex);
    // Serial.print(",");
    // Serial.println(pwm_x);

    int loop_end = micros();
    int delay_time  = Ts - (loop_end - loop_start);
    delayMicroseconds(delay_time);
  } while (abs(e_prev_x) > threshold_error || abs(e_prev_y) > threshold_error);

  // reset the error and error sum terms
  e_prev_y = 0;
  e_prev_x = 0;
  sumErr_y = 0;
  sumErr_x = 0;

  command_motors(GALVO_MOTOR_X1, GALVO_MOTOR_X2, 0);
  command_motors(GALVO_MOTOR_Y1, GALVO_MOTOR_Y2, 0);
  delay(2000);
  command_motors(GALVO_MOTOR_X1, GALVO_MOTOR_X2, 0);
  command_motors(GALVO_MOTOR_Y1, GALVO_MOTOR_Y2, 0);
  return true;
}

MirrorAngles get_mirror_angles(){
  MirrorAngles angles;
  // read the ADCs
  int x_voltage = analogRead(GALVO_POS_X_R);
  int y_voltage = analogRead(GALVO_POS_Y_R);

  // transform to angle
  angles.anglex = slope_x*x_voltage + offsetx;
  angles.angley = slope_y*y_voltage + offsety;

  return angles;
}

float command_motors(int motor_pin1, int motor_pin2, float u) {
  // convert voltage to PWM
  float command = constrain(u, -pwmMax, pwmMax);
  command = u;

  if (command > threshold) {
    command += minPWM;
    analogWrite(motor_pin1, command);
    digitalWrite(motor_pin2, LOW);
  }

  else if (command < -threshold) {
    command -= minPWM;
    analogWrite(motor_pin2, abs(command));
    digitalWrite(motor_pin1, LOW);
  }

  else {
    analogWrite(motor_pin1, 0);
    analogWrite(motor_pin2, 0);
  }

  return command;
  
}

// this is the main loop
bool project_circle(int duration, float magnitude, float frequency) {

  // Lookup table parameters
  const uint16_t tableSize = 360;  // Number of entries in the sine lookup table
  float sineLookup[tableSize];  // Sine wave lookup table
  unsigned long period_us = 1000000 / frequency;  // Period in microseconds

  // create lookup table
  for (int i = 0; i < tableSize; i++) {
    float angle = (2.0 * PI * float(i)) / tableSize;  // Angle in radians
    sineLookup[i] = magnitude * sin(angle);  // Scaled by amplitude
  }

  // step to zero
  bool start_status = 0;
  start_status = center_mirrors_PID(0.1);

  if (!start_status) {
    return false;
  }
  
  int num_cycles = 1000000*duration/Ts;
  int start_time = micros();

  digitalWrite(LASER_PIN, HIGH);

  // repeat for the appropriate number of cycles
  for (int i=0; i<=num_cycles; i++) {
    int loop_start = micros();

    // Get the reference angle from the lookup table
    unsigned long timeInCycle = (loop_start-start_time) % period_us;  // Time within one period
    int tableIndex = map(timeInCycle, 0, period_us, 0, tableSize - 1);  // Map time to table index

    float refAnglex = sineLookup[tableIndex];
    float refAngley = 0.0;
    if (tableIndex < 270) {
      refAngley = sineLookup[tableIndex+90];
    }
    else {
      refAngley = sineLookup[tableIndex-270];
    }

    MirrorAngles angles = get_mirror_angles();
    float mirrorAnglex = angles.anglex;
    float mirrorAngley = angles.angley;

    // calculate error in radians (controller deals in radians)
    float error_x = (refAnglex - mirrorAnglex);
    float error_y = (refAngley - mirrorAngley);

    // calculate command
    float command_x = circle_control_PID(error_x, true);
    float command_y = circle_control_PID(error_y, false);

    float pwm_x = command_motors(GALVO_MOTOR_X1, GALVO_MOTOR_X2, command_x);
    float pwm_y = command_motors(GALVO_MOTOR_Y1, GALVO_MOTOR_Y2, command_y);
    
    if (loop_start % 500 < 100) {
      Serial.print(mirrorAngley);
      Serial.print(",");
      Serial.print(refAngley);
      Serial.print(",");
      Serial.print(pwm_y);
      Serial.print(",");
      Serial.print(mirrorAnglex);
      Serial.print(",");
      Serial.print(refAnglex);
      Serial.print(",");
      Serial.println(pwm_x);
    }
    int loop_end = micros();
    int delay_time  = Ts - (loop_end - loop_start);
    delayMicroseconds(delay_time);
  }
  e_prev_y = 0;
  e_prev_x = 0;
  sumErr_y = 0;
  sumErr_x = 0;

  digitalWrite(LASER_PIN, LOW);
  bool end_status = center_mirrors_PID(0.1);
  return true;

}

void init_project(){
  pinMode(GALVO_MOTOR_X1, OUTPUT);
  pinMode(GALVO_MOTOR_X2, OUTPUT);
  pinMode(GALVO_MOTOR_Y1, OUTPUT);
  pinMode(GALVO_MOTOR_Y2, OUTPUT);
  pinMode(LASER_PIN, OUTPUT);

  pinMode(GALVO_POS_X_R, INPUT);
  pinMode(GALVO_POS_Y_L, INPUT);
  
  analogWriteResolution(15);
}
