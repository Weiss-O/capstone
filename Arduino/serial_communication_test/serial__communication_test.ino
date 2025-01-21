String command

#define sensorPin A0
#define blueLed 8
#define whiteLed 9
#define redLed 10

void Setup() {
  Serial.begin(9600);
  pinMode(blueLed, OUTPUT);
  pinMode(whiteLed, OUTPUT);
  pinMode(redLed, OUTPUT);
}

void loop(){
    int reading = analogRead(sensorPin);
    float voltage = reading * 5.0;
    voltage /= 1024.0;
    float temperatureC = (voltage - 0.5) * 100;
    float temperatureF = (temperatureC * 9.0 / 5.0) + 32.0;

    Serial.println(temperatureF);
    if (Serial.available()) {
        command = Serial.readSringUntil('\n');
        command.trim();
        if (command == "blue") {
            digitalWrite(blueLed, HIGH);
            digitalWrite(whiteLed, LOW);
            digitalWrite(redLed, LOW);
        } else if (command == "white") {
            digitalWrite(blueLed, LOW);
            digitalWrite(whiteLed, HIGH);
            digitalWrite(redLed, LOW);
        } else if (command == "red") {
            digitalWrite(blueLed, LOW);
            digitalWrite(whiteLed, LOW);
            digitalWrite(redLed, HIGH);
        } else if (command == "off") {
            digitalWrite(blueLed, LOW);
            digitalWrite(whiteLed, LOW);
            digitalWrite(redLed, LOW);
        } else {digitalWrite(blueLed, HIGH);
            digitalWrite(whiteLed, HIGH);
            digitalWrite(redLed, HIGH); 
        }
    }
    delay(1000);
}
