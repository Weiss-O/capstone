#ifndef PROJECT_H
#define PROJECT_H

void init_project();

struct MirrorAngles {
    float anglex;
    float angley;
};

float circle_control_PID(float u_new, bool is_x);
bool center_mirrors_PID(float threshold_error);
void calibrate_galvo();
MirrorAngles get_mirror_angles();
float command_motors(int motor_pin1, int motor_pin2, float u);
bool project_circle(int duration, float magnitude, float frequency);

void laser_on();
void laser_off();

#endif
