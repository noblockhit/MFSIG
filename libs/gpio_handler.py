import time
import RPi
import RPi.GPIO as GPIO

# 400 schritte sind pi mal daumen eine Umdrehung


class Motor:
    def __init__(self, pins):
        GPIO.setmode(GPIO.BCM)
        for pin in pins:
            GPIO.setup(pin, GPIO.OUT)

        self.step = 0
        self.pins = pins
        self.pins_dict = {pin: False for pin in pins}
        self.pin_on(self.pins[0])

    def pin_on(self, pin):
        self.pins_dict[pin] = True
        GPIO.output(pin, GPIO.HIGH)

    def pin_off(self, pin):
        self.pins_dict[pin] = False
        GPIO.output(pin, GPIO.LOW)

    def step_forward(self):
        self.step += 1
        if self.step == len(self.pins) * 2:
            self.step = 0

        if self.step % 2 == 0:
            self.pin_on(self.pins[self.step//2])
            self.pin_off(self.pins[(self.step//2-1) % len(self.pins)])
        else:
            self.pin_on(self.pins[(self.step//2+1) % len(self.pins)])

    def step_backward(self):
        self.step -= 1
        if self.step < 0:
            self.step = len(self.pins)*2-1

        if self.step % 2 == 0:
            self.pin_on(self.pins[self.step//2])
            self.pin_off(self.pins[(self.step//2+1) % len(self.pins)])
        else:
            self.pin_on(self.pins[(self.step//2) % len(self.pins)])

    def cleanup(self):
        for pin in self.pins:
            GPIO.output(pin, GPIO.LOW)
        GPIO.cleanup()
        
        
    def calibrate(self):
        for _ in self.pins:
            self.step_forward()
            time.sleep(.3)
            
        for _ in self.pins:
            self.step_backward()
            time.sleep(.3)
        


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
            print(i)
            m.step_backward()
            time.sleep(.01)
            print([int(i) for i in m.pins_dict.values()])

    except:
        import traceback
        traceback.print_exc()
        pass
    finally:
        m.cleanup()