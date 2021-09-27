######################################
# con_detect에 넣기전에 코드 실험
######################################
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import cv2
import time
from influxdb import InfluxDBClient
from copy import deepcopy
import pandas as pd
import serial


# influxdb client 생성
def get_ifdb(db, host='180.70.53.4', port=11334, user='root', passwd='root'):
    # client 객체 생성, 해당 객체는 influxdb에 연결하기 위한 정보를 포함함
    client = InfluxDBClient(host, port, user, passwd, db)

    try:
        # db 기반의 클라이언트 생성
        client.create_database(db)
        print('success')
    except:
        print('failed')
        pass
    return client


def my_test(ifdb, good, bad):
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
		"fields": {
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
	ifdb.write_points(json_body)
	result = ifdb.query('select * from %s' % tablename)


def detect_and_predict_mask(frame, faceNet, maskNet):
	# 프레임의 치수 잡기
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))

	# 얼굴 감지
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# 얼굴 목록, 해당 위치 및 얼굴 마스크 네트워크의 예측 목록을 초기화
	faces = []
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
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# 얼굴만 나왔을때 예측
	if len(faces) > 0:

		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	return (locs, preds)


# 외부 카메라와 연결시 인풋랙으로 인한 꺼짐현상 있어서 강제로 값을 넣어줌
mask = 0
withoutMask = 0
# influxdb에 성공률 넣기
product = 0
good = 0

# 오류 검출을 위한 코드
pro_cnt = 0
mask_cnt = 0
withoutmask_cnt = 0
prev_time = 0
FPS = 10
arduino = serial.Serial('COM4', 9600)
# cv2 (DNN)뉴런 모듈 사용
# 이게 mask_detector.model을 한층 더 구별하기 쉬워짐
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

maskNet = load_model("mask_detector.model")

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

mydb = get_ifdb(db='success_rate')

while True:
	line = arduino.readline()
	# 프레임 조절할꺼임
	frame = vs.read()
	ret = vs.read()
	current_time = time.time() - prev_time
	if (ret is True) and (current_time > 1. / FPS):

		prev_time = time.time()

		cv2.imshow('VideoCapture', frame)

		if cv2.waitKey(1) > 0:
			break
	frame = imutils.resize(frame, width=400)
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	for (box, pred) in zip(locs, preds):

		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# 오류 검출을 위한 count
	mask_cnt = mask_cnt + 1

	# 이부분에 초음파 센서 시리얼 넘버 읽기
	count = float(line[0:4].decode())
	# 시리얼 읽어서 하나씩 더하기 여기에 시리얼 값 변경해줘야 됨
	if count <= 150:
		pro_cnt = pro_cnt + 1

	# 장비정지
	if key == ord("q"):
		bad = product - good
		my_test(mydb, good, bad)
		data = {"product": product, "good_count": good, "bad_count": bad, "good_rate": good / product * 100,
				"bad_rate": bad / product * 100}
		df = pd.DataFrame(data, columns=[product, good, bad, good / product * 100, bad / product * 100])
		df.to_csv('success_rate.csv', mode='a', index=False, encoding='cp949')
		break
# 물건이 없는 상태로 계속 있으면 기기 정지
	elif withoutMask > 0.1 and mask < 0.9998:
		withoutmask_cnt = withoutmask_cnt + 1
		if pro_cnt == 20:
			print('오류')
			product = product + 1
			pro_cnt = 0
			withoutmask_cnt = 0
			arduino.write(b'2\n')
			time.sleep(2)

		elif withoutmask_cnt > 200 and count > 600:
			print('물건이 없습니다. 기기를 정지합니다')
			withoutmask_cnt = 0
			bad = product - good
			my_test(mydb, good, bad)
			data = {"product": product, "good_count": good, "bad_count": bad, "good_rate": good/product * 100, "bad_rate": bad/product * 100}
			df = pd.DataFrame(data, columns=[product, good, bad])
			df.to_csv('success_rate.csv', mode='a', index=False, encoding='cp949')
			break


	elif mask >= 0.9998:
		# 성공 횟수 +=1
		if pro_cnt == 20:
			print('정상')
			product = product + 1
			pro_cnt = 0
			withoutmask_cnt = 0
			good = good + 1
			arduino.write(b'1\n')
			time.sleep(2)

cv2.destroyAllWindows()
vs.stop()
