
#include <Servo.h>



Servo myservoH;  // create servo object to control a servo

int posH = 0;



void setup() {

  myservoH.attach(8);  // 서보모터를 쉴드의 8번에 연결한다.

  Serial.begin(9600);

}



void loop() {

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

    }

    if (posH<0) {

      posH += 180;

    }

    if (posH>180) {

      posH -= 180;

    }


    myservoH.write(posH);


  }

} 
