// DHT11을 적외선 온도 센서로 바꿔주기

#include "DHT.h"
#define DHTPIN 3 
#define DHTTYPE DHT11   // DHT 11


int buzzerPin = 13;  //이부분을 능동부저로 바꿀꺼임
DHT dht(DHTPIN, DHTTYPE);

void setup() {
  Serial.begin(9600);
  pinMode(buzzerPin, OUTPUT);
  dht.begin();
}

void loop() {
  float t = dht.readTemperature();

  while(Serial.available() > 0){
    long value = Serial.parseInt();

    switch (value){
      case 1:
      digitalWrite(buzzerPin, HIGH);
      break;

      case 2:
      digitalWrite(buzzerPin, LOW);
      break;
    }
  }
  Serial.println(t);
}
