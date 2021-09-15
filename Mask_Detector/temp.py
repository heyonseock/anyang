import serial
import time

arduino = serial.Serial('COM5', 9600)

while True:
    data = arduino.readline()
    a = data[0:5].decode()
    a = float(a)
    print(a)