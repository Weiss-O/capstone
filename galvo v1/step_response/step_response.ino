// add after measurement. This is the range of the photodiodes
const int topMax = 708;
const int bottomMax = 693;
const int topMin = 60;
const int bottomMin = 54;

// controller gains
const float kp = 0.15;
const int ki = 0;

// max angular range from -theta to +theta
const float angleRange = 22.5;

// slope and offset values for converting ADC voltage (0-1023) into angle (11.25 to -11.25) in degrees
const float slope = angleRange/((topMax - bottomMin) - (topMin - bottomMax));
const float angleOffset = (angleRange/2) - slope*(topMax - bottomMin);

float integralErrorTotal = 0;
unsigned long previousTime = 0;

const int directionPin1 = 7;
const int directionPin2 = 8;
const int speedPin = 6;

void setup() {
  // Initialize the serial communication at a higher baud rate
  Serial.begin(115200);

  // Configure the analog pins A0 and A1 as input
  pinMode(A0, INPUT); // top photodiode
  pinMode(A1, INPUT); // bottom photodiode

  pinMode(directionPin1, OUTPUT);
  pinMode(directionPin2, OUTPUT);
  pinMode(speedPin, OUTPUT);
 }

void loop() {
  // Record the current time in microseconds
  unsigned long currentTime = micros();
  
  // generate reference signal
  float A = 2; // reference angle amplitude (Degrees)
  float f = 0.1; // reference angle frequency (Hz)

  // for sine wave input
  float refAngle = A * sin((f *2.0 * PI * float(currentTime)) / float(1000000));
  
  // for a step input. Comment this out if you want an actual sine wave
  /*if(refAngle > 0){
    refAngle = A;
  }
  else{
    refAngle = 0;
  }*/

  // Read analog values from A0 and A1
  int analogValueA0 = analogRead(A0);
  int analogValueA1 = analogRead(A1);
  
  // Calculate the angle based on difference
  int analogDiff = analogValueA0 - analogValueA1;
  float mirrorAngle = slope*analogDiff + angleOffset;

  // generate the error term
  float error = refAngle - mirrorAngle;

  // add the error to the total error counter
  integralErrorTotal += (error * (currentTime - previousTime) * (1/1000000));

  // generate control signal using error, total error and gains
  float commandSignal = (kp * error) + ki * (integralErrorTotal);

  if(commandSignal < -255){
    commandSignal = -255;
  }
  else if(commandSignal > 255){
    commandSignal = 255;
  }

  // send commands to the l298n
  if(commandSignal < 0){
    digitalWrite(directionPin1, LOW);
    digitalWrite(directionPin2, HIGH);
  }
  else{
    digitalWrite(directionPin1, HIGH);
    digitalWrite(directionPin2, LOW);
  }
  digitalWrite(speedPin, abs(commandSignal));

  //prints in the following format
  // Current time [us], Angle (measured), Angle (reference), Error, Total integral error, Command Signal
  Serial.print(currentTime);
  Serial.print(",");
  Serial.print(mirrorAngle);
  Serial.print(",");
  Serial.print(refAngle);
  Serial.print(",");
  Serial.print(error);
  Serial.print(",");
  Serial.print(integralErrorTotal);
  Serial.print(",");
  Serial.println(commandSignal);
  
  previousTime = currentTime;
  
  }
