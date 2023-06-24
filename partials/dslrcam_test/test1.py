
import RPi
import RPi.GPIO as GPIO
from time import sleep



GPIO.setmode(GPIO.BCM)

GPIO.setup(26, GPIO.OUT)

try:
    while True:
        sleep(5)
        print("on")
        GPIO.output(26, GPIO.HIGH)

        sleep(5)
        print("off")
        GPIO.output(26, GPIO.LOW)

except KeyboardInterrupt:
    pass
finally:
    GPIO.cleanup()
