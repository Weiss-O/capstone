#ifndef STEPPER_POINT_H
#define STEPPER_POINT_H

void init_stepper();
bool home_stepper();
bool point_steppers(int tilt_steps, int pan_steps);

#endif
