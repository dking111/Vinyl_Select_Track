import RPi.GPIO as GPIO 
from main import main

def button_callback(channel):
    GPIO.output(GPIO.HIGH)
    print("Starting Main")
    GPIO.output(GPIO.LOW)
    return main(verbose=False)


GPIO.setwarnings(False) 
GPIO.setmode(GPIO.BOARD)

#button
GPIO.setup(11, GPIO.IN, pull_up_down=GPIO.PUD_DOWN) 
#ring light
GPIO.setup(12,GPIO.OUT)

#event detection
GPIO.add_event_detect(10,GPIO.RISING,callback=button_callback,bouncetime=1000) 
message = input("Press enter to quit\n\n") 
GPIO.cleanup() #

