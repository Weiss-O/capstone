#include "project.h"
#include "config.h"
#include <Arduino.h>

const float pwmFrequency = 1000;

const float kp_x = 700.0;
const float ki_x = 0.1;
const float kd_x = 150;
const float x_scale = 0.75;

const float kp_y = 800.0;
const float ki_y = 0.1;
const float kd_y = 150;

const float kp_c = 60;
const float ki_c = 0.05;
const float kd_c = 0;

float sumErr_x = 0.0;
float e_prev_x = 0.0;

float sumErr_y = 0.0;
float e_prev_y = 0.0;

const float Ts = 250; // sample time in micros
const int minPWM = 1000;
const int pwmMax = 32767;
const int threshold = 180;

float max_y = 732.0;
float min_y = 282.0;
float max_x = 600.0;
float min_x = 385.0;

float slope_y = 54.0/(max_y - min_y);
float slope_x = 54.0/(max_x - min_x);
float offsetx = -27 - (slope_x*min_x)+1;
float offsety = -27 - (slope_y*min_y);

const int x_galvo_pin = GALVO_POS_X_R;
const int y_galvo_pin = GALVO_POS_Y_R;

void calibrate_galvo(){
  int commandSpeed = 4000;
  // write motors to 3000 in one direction
  command_motors(GALVO_MOTOR_X1, GALVO_MOTOR_X2, commandSpeed);
  command_motors(GALVO_MOTOR_Y1, GALVO_MOTOR_Y2, commandSpeed);

  // hold for one second
  delay(1000);
  // record the value here
  float x1 = analogRead(x_galvo_pin);
  float y1 = analogRead(y_galvo_pin);

  // write motors to 3000 in one direction
  command_motors(GALVO_MOTOR_X1, GALVO_MOTOR_X2, 0);
  command_motors(GALVO_MOTOR_Y1, GALVO_MOTOR_Y2, 0);

  delay(1000);

  // write motors to 3000 in other direction
  command_motors(GALVO_MOTOR_X1, GALVO_MOTOR_X2, -commandSpeed);
  command_motors(GALVO_MOTOR_Y1, GALVO_MOTOR_Y2, -commandSpeed);

  // hold for one second
  delay(1000);

  // record the value here
  float x2 = analogRead(x_galvo_pin);
  float y2 = analogRead(y_galvo_pin);

  command_motors(GALVO_MOTOR_X1, GALVO_MOTOR_X2, 0);
  command_motors(GALVO_MOTOR_Y1, GALVO_MOTOR_Y2, 0);

  delay(1000);

  // check each one to see what is larger and what is smaller
  if (x2 > x1) {max_x = x2; min_x = x1;}
  else {max_x = x1; min_x = x2;}
  if (y2 > y1) {max_y = y2; min_y = y1;}
  else {max_y = y1; min_y = y2;}

  Serial.print("max_x: ");
  Serial.print(max_x);
  Serial.print(" min_x: ");
  Serial.print(min_x);
  Serial.print(" max_y: ");
  Serial.print(max_y);
  Serial.print(" min_y: ");
  Serial.println(min_y);

  if (center_mirrors_PID(0.75)) {Serial.println("finished centering succesfully");}
  else {Serial.println("Fowen");}
  
  return;
}

float circle_control_PID(float e_new, bool is_x) {
  if (is_x) {
    sumErr_x += e_new;
    float dErr = (e_new - e_prev_x)/Ts;
    e_prev_x = e_new;
    // Serial.println(dErr * kd_x);
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
  int num_cycles = 0;
  int run_time = 0;

  do {
    int loop_start = micros();
    run_time = loop_start-start_time;

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
    // Serial.print(pwm_x);
    // Serial.print(",");
    // Serial.print(mirrorAngley);
    // Serial.print(",");
    // Serial.println(pwm_y);

    int loop_end = micros();
    int delay_time  = Ts - (loop_end - loop_start);
    delayMicroseconds(delay_time);

    // Check that the error has been low for a while
    if (abs(e_prev_y) < threshold_error && abs(e_prev_x) < threshold_error) {
      num_cycles++;
      // Serial.println(num_cycles);
    }
    else {
      num_cycles = 0;
    }

  } while (num_cycles < 100 && run_time < 5000000);

  // reset the error and error sum terms
  e_prev_y = 0;
  e_prev_x = 0;
  sumErr_y = 0;
  sumErr_x = 0;

  command_motors(GALVO_MOTOR_X1, GALVO_MOTOR_X2, 0);
  command_motors(GALVO_MOTOR_Y1, GALVO_MOTOR_Y2, 0);
  delay(500);
  command_motors(GALVO_MOTOR_X1, GALVO_MOTOR_X2, 0);
  command_motors(GALVO_MOTOR_Y1, GALVO_MOTOR_Y2, 0);

  MirrorAngles temp = get_mirror_angles();

  // Serial.print(temp.anglex);
  // Serial.print(" , ");
  // Serial.println(temp.angley);

  // Serial.print(num_cycles);
  // Serial.print(" , ");
  // Serial.println(run_time);

  if (run_time < 5000000) {
    return true;
  }
  else {return false;}
}

MirrorAngles get_mirror_angles(){
  MirrorAngles angles;
  // read the ADCs
  int x_voltage = analogRead(x_galvo_pin);
  int y_voltage = analogRead(y_galvo_pin);

  // transform to angle
  angles.anglex = slope_x*x_voltage + offsetx;
  angles.angley = slope_y*y_voltage + offsety;

  return angles;
}

float command_motors(int motor_pin1, int motor_pin2, float u) {
  // convert voltage to PWM
  float command = constrain(u, -pwmMax, pwmMax);

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
  Serial.println("Starting zero");
  bool start_status = 0;
  start_status = center_mirrors_PID(0.75);
  Serial.println("done zero");

  if (!start_status) {
    Serial.println("failed to center");
    return false;
  }
  
  unsigned long num_samples = 1000000*duration/Ts;
  unsigned long start_time = micros();

  digitalWrite(LASER_PIN, HIGH);

  // repeat for the appropriate number of cycles
  for (int i=0; i<=num_samples; i++) {
    unsigned long loop_start = micros();

    // Get the reference angle from the lookup table
    unsigned long timeInCycle = (loop_start-start_time) % period_us;  // Time within one period

    int tableIndex = map(timeInCycle, 0, period_us, 0, tableSize);  // Map time to table index

    float refAnglex = x_scale*sineLookup[tableIndex];
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
    
    // if (i % 10 == 1) {
    //   Serial.print(loop_start-start_time);
    //   Serial.print(",");
    //   Serial.print(mirrorAngley);
    //   Serial.print(",");
    //   Serial.print(refAngley);
    //   Serial.print(",");
    //   Serial.print(pwm_y);
    //   Serial.print(",");
    //   Serial.print(mirrorAnglex);
    //   Serial.print(",");
    //   Serial.print(refAnglex);
    //   Serial.print(",");
    //   Serial.println(pwm_x);
    // }
    int loop_end = micros();
    int delay_time  = Ts - (loop_end - loop_start);
    delayMicroseconds(delay_time);
  }
  e_prev_y = 0;
  e_prev_x = 0;
  sumErr_y = 0;
  sumErr_x = 0;

  digitalWrite(LASER_PIN, LOW);
  bool end_status = center_mirrors_PID(0.75);
  return true;
}

void laser_on() {
  MirrorAngles temp = get_mirror_angles();
  digitalWrite(LASER_PIN, HIGH);
  Serial.print("Laser ON: ");
  Serial.print(temp.anglex);
  Serial.print(",");
  Serial.println(temp.angley);
  return;
}

void laser_off() {
  MirrorAngles temp = get_mirror_angles();
  digitalWrite(LASER_PIN, LOW);
  Serial.print("Laser OFF: ");
  Serial.print(temp.anglex);
  Serial.print(",");
  Serial.println(temp.angley);
  return;
}

void init_project(){
  pinMode(GALVO_MOTOR_X1, OUTPUT);
  pinMode(GALVO_MOTOR_X2, OUTPUT);
  pinMode(GALVO_MOTOR_Y1, OUTPUT);
  pinMode(GALVO_MOTOR_Y2, OUTPUT);
  pinMode(LASER_PIN, OUTPUT);

  pinMode(x_galvo_pin, INPUT);
  pinMode(y_galvo_pin, INPUT);

  analogWriteFrequency(GALVO_MOTOR_X1, pwmFrequency);
  analogWriteFrequency(GALVO_MOTOR_X2, pwmFrequency);
  analogWriteFrequency(GALVO_MOTOR_Y1, pwmFrequency);
  analogWriteFrequency(GALVO_MOTOR_Y2, pwmFrequency);
  
  analogWriteResolution(15);

  //calibrate_galvo();
}
