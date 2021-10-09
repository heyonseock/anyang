# 원본출처 : https://foreverhappiness.tistory.com/62

# -------------------------------------------------------------------------------------------
# 구동방식
# 1. main함수(__main__) 시작
# 2. main함수 내의 do_test 함수 작동
# 	(do_test)
# 	-1. arduino 시리얼 데이터 확인
# 	-2. mydb란 변수명으로 db 생성(get_ifdb 함수 실행)
# 		(get_ifdb)
# 		--1. 기입된 이름의 db 검색 or 생성
# 		--2. db 반환
# 	-3. 수집된 데이터 평가
# 	-4. my_test함수 실행
# 		(my_test)
# 		--1. json_body 이름을 가진 빈 리스트 생성
# 		--2. point라는 이름의 템플릿 생성
# 		--3. 수집된 데이터를 np(point의 딥카피버전)에 저장
# 		--4. json_body에 np를 스택
# 		--5. db에 json_body 저장
# 		--6. query를 통해 저장된 값 확인
#   -5. 이후 for문을 통해 -1로 회귀
# -------------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------------
# 라이브러리 선언
from datetime import datetime
import pprint
import time
from influxdb import InfluxDBClient
from copy import deepcopy
import serial
# -------------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------------
# 연결된 아두이노 포트 기입
device = 'COM3'
# -------------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------------
# influxdb 클라이언트 생성 및 조회 함수
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
# -------------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------------
# 아두이노에서 받은 데이터를 influxdb 클라이언트에 저장 및 터미널에 데이터 출력
def my_test(ifdb, t, h):
    # json_body라는 이름의 세이브포인트(리스트) 생성
    json_body = []
    tablename = 'my_table'
    temp = t
    humi = h

    # point라는 이름의 딕셔너리(key와 value 쌍을 가지는 자료형) 생성
    point = {
        "measurement": tablename,
        "fields": {
            "temp": temp,
            "humi": humi,
        },
        "time": None,
    }


    # UTC 기준을 한국 표준시로 변경
    dt = datetime.now()
    # 깊은 복사로 객체 복사
    np = deepcopy(point)
    np['time'] = dt
    # 추가값이 저장된 np를 json에 스택 저장
    json_body.append(np)
    time.sleep(1)

    # for문에서 완성된 json_body를 influxdb에 저장함
    ifdb.write_points(json_body)

    # influxdb데이터를 불러와 result에 저장 및 출력
    result = ifdb.query('select * from %s' % tablename)
    pprint.pprint(result.raw)


# -------------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------------
# 메인 함수
def do_test():
    # 연결된 시리얼 포트를 통해 온습도 데이터 저장
    arduino = serial.Serial(device, 9600)
    # mydb라는 변수에 클라이언트 생성
    mydb = get_ifdb(db='test2')
    # 해당 클라이언트로 작업(my_test) 수행
    while True:
        time.sleep(1)
        data = arduino.readline()
        t = float(data[0:5].decode())
        h = float(data[5:10].decode())
        # -----------------------------------------------------------------
        # 수집된 데이터 확인
        if t > 35:
            print(f'현재 온, 습도는 {t}°C, {h}% 이며 너무 덥습니다. 에어컨 ON')
            arduino.write(b'1\n')
        elif 27 < t < 35:
            print(f'현재 온, 습도는 {t}°C, {h}% 이며 적당합니다. 에어컨 OFF')
            arduino.write(b'2\n')
        elif t > 47:
            print(f'화재발생')
            arduino.write(b'0\n')
        if h > 70:
            print(f'현재 온, 습도는 {t}°C, {h}% 이며 너무 습합니다. 팬 ON')
            arduino.write(b'3\n')
        elif 40 < h < 70:
            print(f'현재 온, 습도는 {t}°C, {h}% 이며 적당합니다. 팬 OFF')
            arduino.write(b'4\n')
        elif h < 40:
            print(f'현재 온, 습도는 {t}°C, {h}% 이며 너무 건조합니다. 물을 뿌립니다.')
        # -----------------------------------------------------------------
        # 수집된 데이터를 mydb에 저장하는 함수 실행
        my_test(mydb, t, h)
# -------------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------------
# 시작 함수 실행
if __name__ == '__main__':
    # 메인 함수
    do_test()
# -------------------------------------------------------------------------------------------