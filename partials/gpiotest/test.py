import time 
import RPi
import RPi.GPIO as GPIO


pins = [16,19,20,21]

GPIO.setmode(GPIO.BCM)
for pin in pins:
    GPIO.setup(pin, GPIO.OUT)

try:

    while True:
        for pin in reversed(pins):
            GPIO.output(pin, GPIO.HIGH)
            print("click69420")
            time.sleep(.05)
            
            GPIO.output(pin, GPIO.LOW)
except:
    pass
finally:
    print("cleanupIsGenius")
    GPIO.cleanup()