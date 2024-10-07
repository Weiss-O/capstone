// add after measurement. This is the range of the photodiodes
const int16_t topMax = 708;
const int16_t bottomMax = 693;
const int16_t topMin = 60;
const int16_t bottomMin = 54;

// corresponds to 0.9V motor minimum at 5V supply
const int16_t minPWM = 46;

// controller gains
const float kp = 10;
const float ki = 0;

// max angular range from -theta to +theta
const float angleRange = 22.5;

// slope and offset values for converting ADC voltage (0-1023) into angle (11.25 to -11.25) in degrees
const float slope = angleRange/((topMax - bottomMin) - (topMin - bottomMax));
const float angleOffset = (angleRange/2) - slope*(topMax - bottomMin);

float integralErrorTotal = 0;
unsigned long previousTime = 0;

const uint8_t directionPin1 = 7;
const uint8_t directionPin2 = 8;
const uint8_t speedPin = 6;

// Lookup table parameters
const uint16_t tableSize = 360;  // Number of entries in the sine lookup table
float sineLookup[tableSize];  // Sine wave lookup table
float A = 10;  // Amplitude (Degrees)
float f = 0.2; // Frequency (Hz)
unsigned long period_us;  // Period in microseconds

void setup() {
  // Initialize the serial communication at a higher baud rate
  Serial.begin(115200);

  // Configure the analog pins A0 and A1 as input
  pinMode(A0, INPUT); // top photodiode
  pinMode(A1, INPUT); // bottom photodiode

  pinMode(directionPin1, OUTPUT);
  pinMode(directionPin2, OUTPUT);
  pinMode(speedPin, OUTPUT);

  // Precompute the sine lookup table
  for (int i = 0; i < tableSize; i++) {
      float angle = (2.0 * PI * i) / tableSize;  // Angle in radians
      sineLookup[i] = A * sin(angle);  // Scaled by amplitude
  }

  // Calculate the period in microseconds based on frequency
  period_us = 1000000 / f;  // 1 second (1e6 us) divided by frequency
 }

void loop() {
  // Record the current time in microseconds
  unsigned long currentTime = micros();
  
  // Find the corresponding index in the lookup table
  unsigned long timeInCycle = currentTime % period_us;  // Time within one period
  int tableIndex = map(timeInCycle, 0, period_us, 0, tableSize - 1);  // Map time to table index

  // Get the reference angle from the lookup table
  float refAngle = sineLookup[tableIndex];
  /*if (refAngle < 0){
    refAngle = -5;
  }
  else {
    refAngle = 5;
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
  //integralErrorTotal += (error * (currentTime - previousTime))*0.000001;

  // generate control signal using error, total error and gains
  int16_t commandSignal = (kp * error) + ki * integralErrorTotal;

  // send commands to the l298n
  if(commandSignal < -10){
    digitalWrite(directionPin1, LOW);
    digitalWrite(directionPin2, HIGH);
    commandSignal -= minPWM;
  }
  else if (commandSignal > 10){
    digitalWrite(directionPin1, HIGH);
    digitalWrite(directionPin2, LOW);
    commandSignal += minPWM;
  }
  else {
    digitalWrite(directionPin1, LOW);
    digitalWrite(directionPin2, LOW);
  }
  
  commandSignal = constrain(commandSignal, -255, 255);
  if (abs(commandSignal) < 10){
    commandSignal = 0;
  }
  analogWrite(speedPin, abs(commandSignal));

  //prints in the following format
  // Current time [us], Angle (measured), Angle (reference), Error, Total integral error, Command Signal
  if (currentTime % 10000 < 500) {
      //Serial.print(currentTime);
      //Serial.print(",");
      Serial.print(mirrorAngle);
      Serial.print(",");
      Serial.println(refAngle);
      //Serial.print(",");
      //Serial.println(integralErrorTotal);
      //Serial.print(error);
      //Serial.print(",");
      //Serial.println(commandSignal);
  }
  previousTime = currentTime;
  
  }
