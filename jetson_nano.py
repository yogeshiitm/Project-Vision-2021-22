import serial
import time

ser = serial.Serial('/dev/ttyACM0',9600, timeout=5)


import Jetson.GPIO as GPIO 
from pymongo import MongoClient

client = MongoClient("mongodb://Nishant:testgudipaty@testcluster-shard-00-00.p19xk.mongodb.net:27017,testcluster-shard-00-01.p19xk.mongodb.net:27017,testcluster-shard-00-02.p19xk.mongodb.net:27017/myFirstDatabase?ssl=true&replicaSet=atlas-w9nbtk-shard-0&authSource=admin&retryWrites=true&w=majority")
db = client["vision"]
col = db["motor_data"]

pin1 = 13
pin2 = 19

# Set up the GPIO channel
GPIO.setmode(GPIO.BOARD)  
GPIO.setup(pin1, GPIO.OUT, initial=GPIO.LOW) 
GPIO.setup(pin2, GPIO.OUT, initial=GPIO.LOW) 
GPIO.setwarnings(False)

# Blink the LED
while True:
    time.sleep(2)

    data = col.find_one()
    if(data==None):
        data = {"led_left":0,"led_right":0}
"""
    if(data["led_left"]):
        #GPIO.output(pin1, GPIO.HIGH)
        ser.write(b'On\n')
        print("LED1 is on")
    else:
        #GPIO.output(pin1, GPIO.LOW)
        ser.write(b'Off\n')
        print("LED1 is off")
"""
    if(data["led_right"]):
        #GPIO.output(pin2, GPIO.HIGH)
        ser.write(b'On\n')
        print("LED2 is on")
    else:
        #GPIO.output(pin2, GPIO.LOW)
        ser.write(b'Off\n')
        print("LED2 is off")