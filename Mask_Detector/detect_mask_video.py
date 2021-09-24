#######################################
# 마스크 오류 검출 여기다 온도센서랑
# 아두이노 온도 측정 코드는 delay 없이 값을 넣어야 한다. // 다시 아두이노 부분 delay(100)으로 해줬음
# 안그러면 비디오 출력 과정에서 끊기는 현상을 볼 수 있다.
#######################################
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import serial
import os

# 외부 카메라와 연결시 인풋랙으로 인한 꺼짐현상 있어서 강제로 값을 넣어줌
mask = 0
withoutMask = 0

mask_cnt = 0
withoutmask_cnt = 0
prev_time = 0
FPS = 10
arduino = serial.Serial('COM4', 9600)


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


# cv2 (DNN)뉴런 모듈 사용
# 이게 mask_detector.model을 한층 더 구별하기 쉬워짐
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

maskNet = load_model("mask_detector.model")

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

while True:
	data = arduino.readline()
	temp = data[0:5].decode()
	temp = float(temp)
	# 프레임 조절할꺼임
	frame = vs.read()
	ret = vs.read()
	current_time = time.time() - prev_time
	if (ret is True) and (current_time > 1. / FPS):

		prev_time = time.time()

		cv2.imshow('VideoCapture', frame)

		if cv2.waitKey(1) > 0:
			break
	frame = imutils.resize(frame, width=1024)
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	for (box, pred) in zip(locs, preds):

		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		label = "{}C".format(temp)

		cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# 오류 검출을 위한 count
	if mask > 0.98:
		mask_cnt = mask_cnt + 1
	if withoutMask > 0.95 or 0.8 < mask < 94:
		withoutmask_cnt = withoutmask_cnt + 1

	if key == ord("q"):
		break

	if mask > 0.95:
		if mask_cnt > 20:
			if temp < 38.0:
				print('마스크 착용')
				print(f'{temp}°C')
				print('정상')
				arduino.write(b'2\n')
				#time.sleep(1)
			else:
				print('마스크 착용')
				print(f'{temp}°C')
				print('높음')
				arduino.write(b'1\n')
				#time.sleep(1)
				arduino.write(b'2\n')
			mask_cnt = 0
			withoutmask_cnt = 0

	elif withoutMask > 0.95 or 0.8 < mask < 0.94:
		if withoutmask_cnt > 20:
			print('마스크 미착용')
			if temp < 38.0:
				print(f'{temp}°C')
				print('마스크를 써주십쇼')
				arduino.write(b'1\n')
				#time.sleep(1)
				arduino.write(b'2\n')
			else:
				print(f'{temp}°C')
				print('높음')
				arduino.write(b'1\n')
				#time.sleep(1)
				arduino.write(b'2\n')

			mask_cnt = 0
			withoutmask_cnt = 0

	else:
		print('파악하기 힘듬')

cv2.destroyAllWindows()
vs.stop()


