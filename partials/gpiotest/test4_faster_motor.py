import time

global pins_value
pins = [16,19,20,21]


try:
    import RPi
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)

    def p_on(pin):
        GPIO.output(pin, GPIO.HIGH)
    def p_off(pin):
        GPIO.output(pin, GPIO.LOW)

    for pin in pins:
        GPIO.setup(pin, GPIO.OUT)

except ImportError:
    pins_value = {k: 0 for k in pins}

    def p_on(pin):
        pins_value[pin] = 1


    def p_off(pin):
        pins_value[pin] = 0


sleep_time_after_step = 0


class Motor:
    def __init__(self, pins):
        for pin in pins:
            p_off(pin)

        self.step = 0
        self.pins = pins
        self.pins_dict = {pin: False for pin in pins}
        self.pin_on(self.pins[0])
    
    
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
            
        time.sleep(sleep_time_after_step / 1000)

    def step_backward(self):
        self.step -= 1
        if self.step < 0:
            self.step = len(self.pins)*2-1

        if self.step % 2 == 0:
            self.pin_on(self.pins[self.step//2])
            self.pin_off(self.pins[(self.step//2+1) % len(self.pins)])
        else:
            self.pin_on(self.pins[(self.step//2) % len(self.pins)])
            
        time.sleep(sleep_time_after_step / 1000)



class PrebakedMotor:
    def __init__(self, pins):
        for pin in pins:
            p_off(pin)
        self.pins = pins
        self.bake_instructions()

        p_on(self.pins[0])
        self.step = self.instructuion_length * 2
    

    def bake_instructions(self):
        self.instructions = {}
        for idx in range(len(self.pins)):
            self.instructions[idx*2] = self.pins[(idx+1) % len(self.pins)]
            self.instructions[idx*2 + 1] = self.pins[idx]

        self.instructuion_length = len(self.instructions)



    def step_forward(self):
        start = time.perf_counter_ns()
        if self.step % 2 == 0:
            p_on(self.instructions[self.step % self.instructuion_length])

        if self.step % 2 == 1:
            p_off(self.instructions[self.step % self.instructuion_length])
        
        self.step += 1
        if self.step > self.instructuion_length*4:
            self.step = (self.step % self.instructuion_length) + self.instructuion_length
        time.sleep(sleep_time_after_step / 1000 - (time.perf_counter_ns() - start)*10**-9)

    def step_backward(self):
        start = time.perf_counter_ns()
        if self.step % 2 == 0:
            p_on(self.instructions[(self.step-1) % self.instructuion_length])

        if self.step % 2 == 1:
            p_off(self.instructions[(self.step-1) % self.instructuion_length])
        
        self.step -= 1

        if self.step < self.instructuion_length:
            self.step = self.step + self.instructuion_length
        time.sleep(sleep_time_after_step / 1000 - (time.perf_counter_ns() - start)*10**-9)


import random
total_steps = 100_000

test_instructions = []

for i in range(total_steps):
    test_instructions.append(int(random.getrandbits(1)))


history1 = ""
m1 = Motor(pins)
s = time.perf_counter_ns()
for i in test_instructions:
    if i == 0:
        m1.step_forward()
    else:
        m1.step_backward()
    
    # history1 += f"{pins_value}\n"

print((time.perf_counter_ns() - s) * 10**-9)

history2 = ""
m2 = PrebakedMotor(pins)
s = time.perf_counter_ns()
for i in test_instructions:
    if i == 0:
        m2.step_forward()
    else:
        m2.step_backward()
    
    # history2 += f"{pins_value}\n"

print((time.perf_counter_ns() - s) * 10**-9)

assert history2 == history1


