#include "project.h"
#include "config.h"
#include <Arduino.h>

#define N 3
float y_x[N] = {1};  // Output buffer for x
float u_x[N] = {1};  // Input buffer

float y_y[N] = {0};  // Output buffer for y
float u_y[N] = {0};  // Input buffer

const float kp = 1.0;
const float ki = 0.0;
const float kd = 0.0;

const float kp_c = 1.0;
const float ki_c = 0.0;
const float kd_c = 0.0;

float sumErr_x = 0.0;
float e_prev_x = 0.0;

float sumErr_y = 0.0;
float e_prev_y = 0.0;

const float Ts = 1000; // sample time in micros
const int minPWM = 2000;
const int pwmMax = 32757;
const int threshold = 100;

const float slope_y = 0.135;
const float slope_x = 0.282722513;
const float offsetx = -34;
const float offsety = -66;

// controller coefficients
float b[] = {34.057017, -33.988971};
float a = -0.740818;

float b_center[] = {0.182747, -0.182715};

float circle_control_PID(float e_new, bool is_x) {
  if (iz_x) {
    sumErr_x += e_new * Ts;
    float dErr = (e_new - e_prev_x)/Ts;
    e_prev_x = e_new;
    return kp * e_new + ki * sumErr + kd * dErr;
  }

  else {
    sumErr_y += e_new * Ts;
    float dErr = (e_new - e_prev_y)/Ts;
    e_prev_y = e_new;
    return kp * e_new + ki * sumErr + kd * dErr;
  }
}

bool center_mirrors_PID(float threshold_error) {
  int start_time = micros();
  while (abs(e_prev_x) > threshold_error || abs(e_prev_y) > threshold_error) {
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
    float error_x = (0 - mirrorAnglex)*(3.14159/180);
    float error_y = (0 - mirrorAngley)*(3.14159/180);

    sumErr_y += error_y * Ts;
    sumErr_x += error_x * Ts;

    float dErr_x = (error_x - e_prev_x)/Ts;
    float dErr_y = (error_y - e_prev_y)/Ts;

    // apply PID loop
    float command_x = kp * error_x + ki * sumErr_x + kd * dErr_x;
    float command_y = kp * error_y + ki * sumErr_y + kd * dErr_y;

    float pwm_x = command_motors(GALVO_MOTOR_X1, GALVO_MOTOR_X2, command_x);
    float pwm_y = command_motors(GALVO_MOTOR_Y1, GALVO_MOTOR_Y2, command_y);

    // shift enew to eprev
    e_prev_y = error_y;
    e_prev_x = error_x;

    int loop_end = micros();
    int delay_time  = Ts - (loop_end - loop_start);
    delayMicroseconds(delay_time);
  }

  // reset the error and error sum terms
  e_prev_y = 0;
  e_prev_x = 0;
  sumErr_y = 0;
  sumErr_x = 0;

  command_motors(GALVO_MOTOR_X1, GALVO_MOTOR_X2, 0.0);
  command_motors(GALVO_MOTOR_Y1, GALVO_MOTOR_Y2, 0.0);
  return true;
}

float circle_control(float u_new, bool is_x) {
  if (is_x) {
    // Shift previous inputs and outputs
    u_x[1] = u_x[0];
    y_x[1] = y_x[0];

    u_x[0] = u_new;  // Store the new input

    // Compute output using the difference equation
    y_x[0] = b[0]*u_x[0] + b[1]*u_x[1] - a*y_x[1];

    return y_x[0];  // Return the new control output
  }

  else{
    // Shift previous inputs and outputs
    u_y[1] = u_y[0];
    y_y[1] = y_y[0];

    u_y[0] = u_new;  // Store the new input

    // Compute output using the difference equation
    y_y[0] = b[0]*u_y[0] + b[1]*u_y[1] - a*y_y[1];

    return y_y[0];  // Return the new control output
  }

}

bool center_mirrors(float threshold_error) {
  Serial.println("centering started");
  int start_time = micros();
  while (abs(u_x[0]) > threshold_error || abs(u_y[0]) > threshold_error) {
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
    float error_x = (0 - mirrorAnglex)*(3.14159/180);
    float error_y = (0 - mirrorAngley)*(3.14159/180);

    // Shift previous inputs and outputs
    u_x[2] = u_x[1];
    u_y[2] = u_y[1];
    y_x[2] = y_x[1];
    y_y[2] = y_y[1];

    u_x[1] = u_x[0];
    u_y[1] = u_y[0];
    y_x[1] = y_x[0];
    y_y[1] = y_y[0];

    u_x[0] = error_x;  // Store the new input
    u_y[0] = error_y;  // Store the new input

    // Compute output using the difference equation
    y_y[0] = b_center[0]*u_y[0] + b_center[1]*u_y[1] + y_y[1];
    y_x[0] = 0.0*u_x[0] - 0.021124*u_x[1] - 0*u_x[2] + 0*y_x[1] - 0.0*y_x[2];

    float pwm_x = command_motors(GALVO_MOTOR_X1, GALVO_MOTOR_X2, y_x[0]);
    float pwm_y = command_motors(GALVO_MOTOR_Y1, GALVO_MOTOR_Y2, y_y[0]);

    Serial.print(loop_start);
    Serial.print(",");
    Serial.print(mirrorAnglex);
    Serial.print(",");
    Serial.println(pwm_x);

    int loop_end = micros();
    int delay_time  = Ts - (loop_end - loop_start);
    delayMicroseconds(delay_time);

  }
  command_motors(GALVO_MOTOR_X1, GALVO_MOTOR_X2, 0.0);
  command_motors(GALVO_MOTOR_Y1, GALVO_MOTOR_Y2, 0.0);

  Serial.print(u_x[0]);
  Serial.print(",");
  Serial.println(u_y[0]);
  Serial.println("centering completed");
  return true;
}

MirrorAngles get_mirror_angles(){
  MirrorAngles angles;
  // read the ADCs
  int x_voltage = analogRead(GALVO_POS_X_L);
  int y_voltage = analogRead(GALVO_POS_Y_R);

  // transform to angle
  angles.anglex = slope_x*x_voltage + offsetx;
  angles.angley = slope_y*y_voltage + offsety;

  return angles;
}

float command_motors(int motor_pin1, int motor_pin2, float u) {
  // convert voltage to PWM
  float command = constrain(u, -5.0, 5.0);
  command = command*(float(pwmMax)/5.0);

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
    digitalWrite(motor_pin1, LOW);
    digitalWrite(motor_pin2, LOW);
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
  start_status = center_mirrors_PID(0.05);

  if (!start_status) {
    return false;
  }

  // clear the buffers
  u_x[0] = 0;
  u_x[1] = 0;
  y_x[0] = 0;
  y_x[1] = 0;

  u_y[0] = 0;
  u_y[1] = 0;
  y_y[0] = 0;
  y_y[1] = 0;
  
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
    float refAngley = 0;
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
    float error_x = (refAnglex - mirrorAnglex)*(3.14159/180);
    float error_y = (refAngley - mirrorAngley)*(3.14159/180);

    // calculate command
    float command_x = circle_control_PID(error_x, true);
    float command_y = circle_control_PID(error_y, false);

    float pwm_x = command_motors(GALVO_MOTOR_X1, GALVO_MOTOR_X2, command_x);
    float pwm_y = command_motors(GALVO_MOTOR_Y1, GALVO_MOTOR_Y2, command_y);

    int loop_end = micros();
    int delay_time  = Ts - (loop_end - loop_start);
    delayMicroseconds(delay_time);
    if (loop_start % 500 < 100) {
      // Serial.print(tableIndex);
      // Serial.print(",");
      Serial.print(mirrorAnglex);
      Serial.print(",");
      Serial.print(refAnglex);
      Serial.print(",");
      Serial.println(pwm_x);
    }
  }

  digitalWrite(LASER_PIN, LOW);
  bool end_status = center_mirrors(0.05);
  
  // clear the buffers
  u_x[0] = 1;
  u_x[1] = 0;
  y_x[0] = 0;
  y_x[1] = 0;

  u_y[0] = 1;
  u_y[1] = 0;
  y_y[0] = 0;
  y_y[1] = 0;
  
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
