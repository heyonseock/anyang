 #include <Wire.h>
#include <Adafruit_MLX90614.h>
Adafruit_MLX90614 mlx = Adafruit_MLX90614();
int buzzerPin = 13;  //이부분을 능동부저로 바꿀꺼임

void setup() {
  Serial.begin(9600);
  pinMode(buzzerPin, OUTPUT);
  mlx.begin();
}

void loop() {

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
  Serial.println(mlx.readObjectTempC()); 
  delay(150);
}
