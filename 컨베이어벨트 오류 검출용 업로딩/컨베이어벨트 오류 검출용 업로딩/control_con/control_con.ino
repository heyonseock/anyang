#include <Servo.h>
#define trig 8    // 트리거 핀 선언
#define echo 9    // 에코 핀 선언
#define con1 6    // 컨베이어 제어
#define con2 7    // 컨베이어 제어
Servo myservoH;  // create servo object to control a servo
int posH = 0;

void setup()
{
  Serial.begin(9600);      // 통신속도 9600bps로 시리얼 통신 시작
  myservoH.attach(5);  // 서보모터를 쉴드의 5번에 연결한다.
  pinMode(trig, OUTPUT);    // 트리거 핀 출력으로 선언
  pinMode(echo, INPUT);     // 에코 핀 입력으로 선언
  pinMode(con1, OUTPUT);
  pinMode(con2, OUTPUT);
  digitalWrite(con1, HIGH);
  digitalWrite(con2, LOW);
}
void loop()
{
  long duration, distance;    // 거리 측정을 위한 변수 선언
  // 트리거 핀으로 10us 동안 펄스 출력
  digitalWrite(trig, LOW);        // Trig 핀 Low
  delayMicroseconds(2);            // 2us 딜레이
  digitalWrite(trig, HIGH);    // Trig 핀 High
  delayMicroseconds(10);            // 10us 딜레이
  digitalWrite(trig, LOW);        // Trig 핀 Low
   /*
       에코핀에서 받은 펄스 값을 pulseIn함수를 호출하여
       펄스가 입력될 때까지의 시간을 us단위로 duration에 저장
       pulseln() 함수는 핀에서 펄스(HIGH or LOW)를 읽어서 마이크로초 단위로 반환
  */
  duration = pulseIn(echo, HIGH); 
  /*
       음파의 속도는 초당 340m, 왕복하였으니 나누기 2를하면 170m이고,
       mm단위로 바꾸면 170,000mm.
       duration에 저장된 값은 us단위이므로 1,000,000으로 나누어 주고,
       정리해서 distance에 저장
  */
  distance = duration * 170 / 1000;
  Serial.println(distance); // 거리를 시리얼 모니터에 출력
  while (Serial.available() > 0) {
    long value = Serial.parseInt();
    int backUpH = posH;
    switch (value) {
      case 1:
        posH -= 180;
        break;

      case 2:
        posH += 180;
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
  delay(100);
}
