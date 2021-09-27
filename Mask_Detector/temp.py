# 시리얼 불러오는거 확인용
# 곧 삭제할꺼임
import serial
import time

arduino = serial.Serial('COM4', 9600)

while True:
    data = arduino.readline()
    a = data[0:4].decode()
    a = float(a)
    print(a)