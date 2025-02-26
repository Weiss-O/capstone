void setup() {
  // put your setup code here, to run once:
    Serial.begin(115200);

    pinMode(A0, INPUT); // TL
    pinMode(A1, INPUT); // BL
    pinMode(A2, INPUT); // TR
    pinMode(A3, INPUT); // BR
    pinMode(A4, INPUT); // pot

}

void loop() {
  // put your main code here, to run repeatedly:
  float TL = analogRead(A0);
  float BL = analogRead(A1);
  float TR = analogRead(A2);
  float BR = analogRead(A3);
  float pot = analogRead(A4);

  Serial.print(TL);
  Serial.print(",");
  Serial.print(BL);
  Serial.print(",");
  Serial.print(TR);
  Serial.print(",");
  Serial.print(BR);
  Serial.print(",");
  Serial.println(pot);
}
