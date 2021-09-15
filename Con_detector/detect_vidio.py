#####################
# 컨베이어 with 모터
#####################
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
import serial

On_cnt = 0
Without_cnt = 0
prev_time = 0
FPS = 10
arduino = serial.Serial('COM4', 9600)


def detect_and_predict_con(frame, conNet, detectNet):
    # 프레임의 치수 잡기
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))

    # 감지
    conNet.setInput(blob)
    detections = conNet.forward()
    print(detections.shape)

    # 목록, 해당 위치 및 오류검출 네트워크의 예측 목록을 초기화
    cons = []
    locs = []
    preds = []

    # 탐지 ing
    for i in range(0, detections.shape[2]):
        # 확률 추출
        confidence = detections[0, 0, i, 2]

        # 적은 확률 버리기
        if confidence > 0.5:
            # 객체에 대한 경계 상자의 (x, y) 좌표를 계산
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # 박스 크기
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # 카메라
            con = frame[startY:endY, startX:endX]
            con = cv2.cvtColor(con, cv2.COLOR_BGR2RGB)
            con = cv2.resize(con, (224, 224))
            con = img_to_array(con)
            con = preprocess_input(con)

            cons.append(con)
            locs.append((startX, startY, endX, endY))

    # 예측
    if len(cons) > 0:
        cons = np.array(cons, dtype="float32")
        preds = detectNet.predict(cons, batch_size=32)

    return (locs, preds)


# cv2 (DNN)뉴런 모듈 사용
# 이게 detector.model을 한층 더 구별하기 쉬워짐
prototxtPath = r"detector\deploy.prototxt"
weightsPath = r"detector\res10_300x300_ssd_iter_140000.caffemodel"
conNet = cv2.dnn.readNet(prototxtPath, weightsPath)

detectNet = load_model("detector.model")

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

while True:
    #프레임 조절할꺼임
    frame = vs.read()
    ret = vs.read()
    current_time = time.time() - prev_time
    if (ret is True) and (current_time > 1. / FPS):

        prev_time = time.time()

        cv2.imshow('VideoCapture', frame)

        if cv2.waitKey(1) > 0:
            break
    frame = imutils.resize(frame, width=400)
    (locs, preds) = detect_and_predict_con(frame, conNet, detectNet)

    for (box, pred) in zip(locs, preds):

        (startX, startY, endX, endY) = box
        (On, Without) = pred

        label = "ON" if On > Without else "Without"
        color = (0, 255, 0) if label == "ON" else (0, 0, 255)

        label = "{}: {:.2f}%".format(label, max(On, Without) * 100)

        cv2.putText(frame, label, (startX, startY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

#오류 검출을 위한 count

    On_cnt = On_cnt + 1
    Without_cnt = Without_cnt + 1

    if key == ord("q"):
        break

    elif Without > 0.1 and On < 0.7:
        print('없음')
        arduino.write(b'2\n')
        if Without_cnt > 100:
            print('물건이 없습니다. 기기를 정지합니다')
            Without_cnt = 0
            break


    elif On > 0.9999998:
        print('정상')

        arduino.write(b'1\n')
        Without_cnt = 0
        time.sleep(2)

cv2.destroyAllWindows()
vs.stop()



