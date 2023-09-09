import time
import atexit
# 400 schritte sind pi mal daumen eine Umdrehung
from .state import State, abs_camera_type, abs_motor_type


def p_on(pin):
    if State.GPIO_default_on:
        GPIO.output(pin, GPIO.LOW)
    else:
        GPIO.output(pin, GPIO.HIGH)


def p_off(pin):
    if State.GPIO_default_on:
        GPIO.output(pin, GPIO.HIGH)
    else:
        GPIO.output(pin, GPIO.LOW)



class Motor(abs_motor_type):
    def __init__(self, pins):
        for pin in pins:
            GPIO.setup(pin, GPIO.OUT)
            p_off(pin)

        self.step = 0
        self.pins = pins
        self.pins_dict = {pin: False for pin in pins}
        self.pin_on(self.pins[0])
        
        atexit.register(self.cleanup)
    
    def pin_on(self, pin):
        self.pins_dict[pin] = True
        p_on(pin)

    def pin_off(self, pin):
        self.pins_dict[pin] = False
        p_off(pin)

    def step_forward(self):
        self.step += 1
        if self.step == len(self.pins) * 2:
            self.step = 0

        if self.step % 2 == 0:
            self.pin_on(self.pins[self.step//2])
            self.pin_off(self.pins[(self.step//2-1) % len(self.pins)])
        else:
            self.pin_on(self.pins[(self.step//2+1) % len(self.pins)])
            
        time.sleep(State.sleep_time_after_step)

    def step_backward(self):
        self.step -= 1
        if self.step < 0:
            self.step = len(self.pins)*2-1

        if self.step % 2 == 0:
            self.pin_on(self.pins[self.step//2])
            self.pin_off(self.pins[(self.step//2+1) % len(self.pins)])
        else:
            self.pin_on(self.pins[(self.step//2) % len(self.pins)])
            
        time.sleep(State.sleep_time_after_step)


    def cleanup(self):
        for pin in self.pins:
            p_off(pin)
        GPIO.cleanup()
        
        
    def calibrate(self):
        for _ in self.pins:
            self.step_forward()
            time.sleep(.3)
            
        for _ in self.pins:
            self.step_backward()
            time.sleep(.3)
        

class Camera(abs_camera_type):
    def __init__(self, bcm_pin_number):
        self.pin = bcm_pin_number
        GPIO.setup(self.pin, GPIO.OUT)
        p_off(self.pin)

    def Close(self):
        p_off(self.pin)

    def Snap(self, *_):
        p_on(self.pin)
        time.sleep(2)
        p_off(self.pin)
        time.sleep(State.digi_cam_delay)
        State.progress()


import RPi
import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BCM)
