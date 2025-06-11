import RPi.GPIO as GPIO 
import os

def button_callback(channel):
    os.system('python main.py')


GPIO.setwarnings(False) 
GPIO.setmode(GPIO.BOARD) 
GPIO.setup(11, GPIO.IN, pull_up_down=GPIO.PUD_DOWN) 
GPIO.add_event_detect(10,GPIO.RISING,callback=button_callback,bouncetime=1000) 
message = input("Press enter to quit\n\n") 
GPIO.cleanup() #

button_callback(1)