import RPi.GPIO as GPIO 
from main import main

def button_callback(channel):
    print("Starting Main")
    return main(verbose=False)


GPIO.setwarnings(False) 
GPIO.setmode(GPIO.BOARD) 
GPIO.setup(11, GPIO.IN, pull_up_down=GPIO.PUD_DOWN) 
GPIO.add_event_detect(10,GPIO.RISING,callback=button_callback,bouncetime=1000) 
message = input("Press enter to quit\n\n") 
GPIO.cleanup() #

