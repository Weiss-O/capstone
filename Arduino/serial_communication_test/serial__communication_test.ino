#include <SoftwareSerial.h>

const int timeout = 5000; // 5 seconds timeout
const char *message = "Hello from Arduino!"; // Message to send to the Raspberry Pi
SoftwareSerial mySerial(0, 1); // RX, TX

void setup() {
  Serial.begin(9600); // Serial communication for the laptop monitor
  mySerial.begin(9600); // Serial communication for the software UART
  Serial.println("Sending message to Raspberry Pi...");
  mySerial.println(message); // Send the message to the Raspberry Pi

  unsigned long startTime = millis(); // Record the start time

  while (millis() - startTime < timeout) {
    if (mySerial.available() > 0) { // Check if a response is received
      String response = mySerial.readString(); // Read the response
      Serial.print("Received response from Raspberry Pi: ");
      Serial.println(response);
      delay(1000); // Wait 1 second before sending the next message
      return; // Exit the loop after receiving a response
    }
  }

  Serial.println("No response received, retrying...");
  delay(1000); // Wait 1 second before retrying
}

int main(){
}
