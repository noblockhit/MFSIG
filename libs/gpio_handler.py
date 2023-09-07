import time
import atexit
# 400 schritte sind pi mal daumen eine Umdrehung
from .state import State


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



class Motor(State.abs_motor_type):
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
            
        time.sleep(0.0005)

    def step_backward(self):
        self.step -= 1
        if self.step < 0:
            self.step = len(self.pins)*2-1

        if self.step % 2 == 0:
            self.pin_on(self.pins[self.step//2])
            self.pin_off(self.pins[(self.step//2+1) % len(self.pins)])
        else:
            self.pin_on(self.pins[(self.step//2) % len(self.pins)])
            
        time.sleep(0.0005)


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
        

class Camera(State.abs_camera_type):
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
        time.sleep(.5)
        State.progress()
        


import RPi
import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BCM)

if __name__ == "__main__":
    try:
        m = Motor([16, 19, 20, 21])
        while True:
            m.calibrate()
        
        m.cleanup()
        exit()
        time.sleep(5)
        
        i = 0
        
        
        while True:
            i += 1
            m.step_backward()
            time.sleep(.01)

    except:
        import traceback
        traceback.print_exc()
    finally:
        m.cleanup()
