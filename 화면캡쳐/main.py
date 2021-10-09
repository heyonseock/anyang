########################################
# 물건 학습을 편하게 시키기 위하여 만든 코드 입니다.
########################################

import cv2
import numpy as np
import time


capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 250)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 350)

con_cnt = 0
on_cnt = 0


# 이미지 처리하기
def preprocessing(frame):
    #frame_fliped = cv2.flip(frame, 1)
    # 사이즈 조정
    size = (224, 224)
    frame_resized = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)

    # 이미지 정규화
    # astype : 속성
    frame_normalized = (frame_resized.astype(np.float32) / 127.0) - 1

    # 이미지 차원 재조정 - 예측을 위해 reshape 해줍니다.
    # keras 모델에 공급할 올바른 모양의 배열 생성
    frame_reshaped = frame_normalized.reshape((1, 224, 224, 3))
    # print(frame_reshaped)
    return frame_reshaped


while True:
    # 한 프레임씩 읽기
    ret, frame = capture.read()
    exit_key = cv2.waitKey(1)
    con_cap_key = cv2.waitKey(1)
    on_cap_key = cv2.waitKey(1)
    preprocessed = preprocessing(frame)
    cv2.imshow("VideoFrame", frame)

    if exit_key == ord("q"):
        break

    # z 누르면 캡쳐(콘테이너 왔을때 해주십쇼)
    if con_cap_key == ord("z"):
        con_cnt = con_cnt + 1
        cv2.imwrite("C:/Users/jpg03/Desktop/anyang/Con_detector/dataset/data/con" + str(con_cnt) + ".jpg", frame)
    # x누르면 캡쳐(물건만 캡쳐해 주십셔)
    if on_cap_key == ord("x"):
        on_cnt = on_cnt + 1
        cv2.imwrite("C:/Users/jpg03/Desktop/anyang/Con_detector/dataset/data/ON" + str(on_cnt) + ".jpg", frame)


cv2.destroyAllWindows()
