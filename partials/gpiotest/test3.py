import time 
import RPi
import RPi.GPIO as GPIO


## 400 schritte sind pi mal daumen eine Umdrehung


pins = [16,19,20,21]
aktivePin = None

GPIO.setmode(GPIO.BCM)
for pin in pins:
    GPIO.setup(pin, GPIO.OUT)


class Motor:
    def __init__(self, pins):
        self.pins = pins
        self.idx = 0
        
        
    def pin_on(self, pin):
        GPIO.output(pin, GPIO.HIGH)
 
 
    def pin_off(self, pin):
        GPIO.output(pin, GPIO.LOW)

    
    def step_forward(self):
        self.idx = (self.idx + 1 + len(self.pins)) % len(self.pins)
        self.set_pins()
    
        
    
    def step_backward(self):
        self.idx = (self.idx - 1 + len(self.pins)) % len(self.pins)
        self.set_pins()
		
		
    def set_pins(self):
        pins_idx_on = set()
        pins_idx_on.add(self.idx)
        pins_idx_on.add((self.idx + 1) % len(self.pins))
        
        pins_idx_off = set(list(range(len(self.pins))))
        
        pins_idx_off = pins_idx_off - pins_idx_on
		
        print(pins_idx_on, pins_idx_off)

        for on in pins_idx_on:
            self.pin_on(self.pins[on])

        
        for off in pins_idx_off:
            self.pin_off(self.pins[off])
			
        
        


try:
    m = Motor(pins)
    i = 0
    while True:
        i += 1
        print(i)
        m.step_backward()
        time.sleep(.25)
    
except:
    import traceback
    traceback.print_exc()
    pass
finally:
    for pin in pins:
        GPIO.output(pin, GPIO.LOW)
    print("cleanupIsGenius")
    GPIO.cleanup()
