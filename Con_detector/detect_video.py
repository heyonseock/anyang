import multiprocessing
import numpy as np
import cv2
import tensorflow.keras as tf
import math
import os
import time
from influxdb import InfluxDBClient
from copy import deepcopy
import pandas as pd
import serial
# 만약 cv2.imshow() 가 안먹힌다면 matplotlib 사용
# import matplotlib.pyplot as plt


def get_ifdb(db, host='180.70.53.4', port=11334, user='root', passwd='root'):
    client = InfluxDBClient(host, port, user, passwd, db)

    try:
        client.create_database(db)
        print('success')
    except:
        print('failed')
        pass
    return client


def my_test(ifdb, good, bad, product):
    json_body = []
    tablename = 'my_table'
    fieldname = 'my_field'
    good_count = good
    bad_count = bad
    good_rate = good/product * 100
    bad_rate = bad/product * 100

    point = {
        "measurement": tablename,
        "tags": {
            "Success_rate": "1st"
        },
        "fields":{
            fieldname: 0,
            "good_count": good_count,
            "bad_count": bad_count,
            "good_rate": good_rate,
            "bad_rate": bad_rate,
        },
        "time": None,
    }
    np = deepcopy(point)
    json_body.append(np)
    time.sleep(1)
    ifdb.write_popints(json_body)
    result = ifdb.query('select * from %s' % tablename)


On = 0
Without = 0

# 시리얼 포트
arduino = serial.Serial('COM5', 9600)

def main():
    product = 0
    good = 0
    pro_cnt = 0
    On_cnt = 0
    Without_cnt = 0
    line = arduino.readline()

    # 이부분에 초음파 센서 시리얼 넘버 읽기
    count = float(line[0:4].decode())

    # 시리얼 읽어서 하나씩 더하기 여기에 시리얼 값 변경해줘야 됨
    if count <= 150:
        pro_cnt = pro_cnt + 1
    if count > 600:
        Without_cnt = Without_cnt + 1
    # 라벨 읽기
    labels_path = "C:/Users/lee37/Desktop/anyang/Con_detector/model/labels.txt"
    labelsfile = open(labels_path, 'r')

    # 클래스 선언
    classes = []
    line = labelsfile.readline()
    while line:
        classes.append(line.split(' ', 1)[1].rstrip())
        line = labelsfile.readline()
    # 라벨파일 닫기
    labelsfile.close()

    # h5 모델 불러오기
    model_path = "C:/Users/lee37/Desktop/anyang/Con_detector/model/detector.h5"
    model = tf.models.load_model(model_path, compile=False)

    # 비디오
    cap = cv2.VideoCapture(0)

    # 비디오 크기
    frameWidth = 2000
    frameHeight = 720

    # set width and height in pixels
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)
    cap.set(cv2.CAP_PROP_GAIN, 0)

    # influxdb db
    mydb = get_ifdb(db='success_rate')

    while True:
        key = cv2.waitKey(1) & 0xFF
        np.set_printoptions(suppress=True)

        # 케라스 모델에 공급할 올바른 모양의 배열을 작성
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

        # 이미지 캡쳐
        check, frame = cap.read()

        # h5 모델 사용하기 위해 사각형으로 자르기
        margin = int(((frameWidth-frameHeight)/2))
        square_frame = frame[0:frameHeight, margin:margin + frameHeight]
        # 244*244로 크기 조정
        resized_img = cv2.resize(square_frame, (224, 224))
        # 모델을 위해 색 조정
        model_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

        # 이미지 numpy배열로 만들기
        image_array = np.asarray(model_img)
        # 이미지 재조정
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        # 배열에서 이미지 불러오기
        data[0] = normalized_image_array

        # 예측 실행
        predictions = model.predict(data)

        # 확률은 90%.
        conf_threshold = 90
        confidence = []
        conf_label = ""
        threshold_class = ""
        # 밑에 클래스 정보들 뜨게하기
        per_line = 2
        bordered_frame = cv2.copyMakeBorder(
            square_frame,
            top=0,
            bottom=30 + 15*math.ceil(len(classes)/per_line),
            left=0,
            right=0,
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0]
        )
        # 하나의 클래스마다 정의해주기 이렇게 하면 여러가지 물건들을 학습시키고, 다른 물건 오류 예측도 할 수 있다.
        for i in range(0, len(classes)):
            # 예측 신뢰도를 %로 확장
            confidence.append(int(predictions[0][i]*100))
            # 클래스 숫자에 따라 하단에 텍스트 넣기
            if (i != 0 and not i % per_line):
                cv2.putText(
                    img=bordered_frame,
                    text=conf_label,
                    org=(int(0), int(frameHeight+25+15*math.ceil(i/per_line))),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(255, 255, 255)
                )
                conf_label = ""
            # 확률
            conf_label += classes[i] + ": " + str(confidence[i]) + "%; "
            # 마지막줄
            if (i == (len(classes)-1)):
                cv2.putText(
                    img=bordered_frame,
                    text=conf_label,
                    org=(int(0), int(frameHeight+25+15*math.ceil((i+1)/per_line))),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(255, 255, 255)
                )
                conf_label = ""
            # 정지
            if key == ord("q"):
                bad = product - good
                my_test(mydb, good, bad)
                data = {"product": product, "good_count": good, "bad_count": bad, "good_rate": good / product * 100,
                        "bad_rate": bad / product * 100}
                df = pd.DataFrame(data, columns=[product, good, bad, good / product * 100, bad / product * 100])
                df.to_csv('success_rate.csv', mode='a', index=False, encoding='cp949')
                arduino.write(b'3\n')
                break
            # 성공
            if confidence[0] > conf_threshold:
                threshold_class = classes[i]
                if pro_cnt == 20:
                    product = product + 1
                    pro_cnt = 0
                    Without_cnt = 0
                    good = good + 1
            #         # 아두이노 모터 제어
                    arduino.write(b'1\n')
            # # 아무것도 없을때
            elif Without_cnt > 200 and count > 600 and confidence[1] > conf_threshold:
                print('물건이 없습니다. 기기를 정지합니다')
                Without_cnt = 0
                bad = product - good
                my_test(mydb, good, bad)
                data = {"product": product, "good_count": good, "bad_count": bad, "good_rate": good / product * 100,
                        "bad_rate": bad / product * 100}
                df = pd.DataFrame(data, columns=[product, good, bad])
                df.to_csv('success_rate.csv', mode='a', index=False, encoding='cp949')
                arduino.write(b'3\n')
                break
            # # 오류
            elif confidence[0] < 20:
                product = product + 1
                pro_cnt = 0
                Without_cnt = 0
                # 아두이노 모터 제어
                arduino.write(b'2\n')
                time.sleep(2)

        # cv2.putText(

        #     img=bordered_frame,
        #     text=threshold_class,
        #     org=(int(0), int(frameHeight+20)),
        #     fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        #     fontScale=0.75,
        #     color=(255, 255, 255)
        # )

        # 비디오 피드 구현
        cv2.imshow("Capturing", bordered_frame)

        cv2.waitKey(10)
        if key == ord("q"):
            break

    # terminate process 1

