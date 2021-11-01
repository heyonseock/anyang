#include <Servo.h>
#define con1 6    // 컨베이어 제어
#define con2 7    // 컨베이어 제어
Servo myservoH;  // create servo object to control a servo
int posH = 0;

void setup()
{
  Serial.begin(9600);      // 통신속도 9600bps로 시리얼 통신 시작
  myservoH.attach(5);  // 서보모터를 쉴드의 5번에 연결한다.

  pinMode(con1, OUTPUT);
  pinMode(con2, OUTPUT);
  digitalWrite(con1, HIGH);
  digitalWrite(con2, LOW);
}
void loop()
{
   /*
       에코핀에서 받은 펄스 값을 pulseIn함수를 호출하여
       펄스가 입력될 때까지의 시간을 us단위로 duration에 저장
       pulseln() 함수는 핀에서 펄스(HIGH or LOW)를 읽어서 마이크로초 단위로 반환
  */

  /*
       음파의 속도는 초당 340m, 왕복하였으니 나누기 2를하면 170m이고,
       mm단위로 바꾸면 170,000mm.
       duration에 저장된 값은 us단위이므로 1,000,000으로 나누어 주고,
       정리해서 distance에 저장
  */

  while (Serial.available() > 0) {
    long value = Serial.parseInt();
    int backUpH = posH;
    switch (value) {
      case 1:
        posH -= 180;
        delay(200);
        break;

      case 2:
        posH += 180;
        delay(200);
        break;
      case 3:
        digitalWrite(con2, HIGH);
      case 4:
        digitalWrite(con2, LOW);
    }

    if (posH<0) {
      posH += 180;
    }

    if (posH>180) {
      posH -= 180;
    }


    myservoH.write(posH);


  }
  delay(500);
}
