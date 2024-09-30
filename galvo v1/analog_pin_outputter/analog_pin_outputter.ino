void setup() {
  // Initialize the serial communication at 9600 baud rate
  Serial.begin(9600);
  
  // Configure the analog pins A0 and A1 as input (not strictly necessary, as analog pins default to input mode)
  pinMode(A0, INPUT);
  pinMode(A1, INPUT);
}

void loop() {
  // Read analog values from A0 and A1
  int analogValueA0 = analogRead(A0);
  int analogValueA1 = analogRead(A1);

  // Send the values over the serial port
  // Serial.print("A0: ");
  Serial.print(analogValueA0);
  Serial.print(",");
  Serial.println(analogValueA1);
  
  // Add a delay for readability; adjust as needed
  delay(1);  // 10ms delay between reads
}
