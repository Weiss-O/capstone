#ifndef CONFIG_H
#define CONFIG_H

//DC motor pins
#define GALVO_MOTOR_X1 0
#define GALVO_MOTOR_X2 1
#define GALVO_MOTOR_Y1 22
#define GALVO_MOTOR_Y2 23

//galvo position feedback pins
#define GALVO_POS_Y_R 14
#define GALVO_POS_Y_L 15
#define GALVO_POS_X_R 16
#define GALVO_POS_X_L 17

//limit switch pins
#define SWITCH_TILT 18
#define SWITCH_PAN 19

//stepper motor pins
#define STEPPER_PAN_1 6
#define STEPPER_PAN_2 7
#define STEPPER_PAN_3 8
#define STEPPER_PAN_4 9

#define STEPPER_TILT_1 2
#define STEPPER_TILT_2 3
#define STEPPER_TILT_3 4
#define STEPPER_TILT_4 5

//trigger pins
#define LASER_PIN 13
#define FAN_PIN 20

// Other constants
#define SERIAL_BAUD_RATE 9600
#define STEPPER_TYPE 4
//Test
#endif
