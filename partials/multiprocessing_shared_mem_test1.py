import multiprocessing
from multiprocessing import shared_memory
import time
import random

def fibonacciGenerator():
    a=0
    b=1
    while True:
        yield b
        a,b= b,a+b


def main_deliver():
    shm_deliver = shared_memory.SharedMemory(name="out1", create=True, size=1024)
    buff = shm_deliver.buf
    gen = fibonacciGenerator()
    while True:
        time.sleep(.5)
        n = next(gen)
        print("put in: ", n)
        buff[0] = n


def main_receiver():
    while True:
        try:
            shm_receiver = shared_memory.SharedMemory(name="out1")
        except FileNotFoundError:
            print("Warning: The shared memory names might not be equal, waiting for other process to show up. Terminate the program if this message persists!")
            time.sleep(1)
        else:
            break
    
    prev = None
    while True:
        curr = shm_receiver.buf[0]
        if curr == prev:
            time.sleep(.5)
            continue
        print(curr)
        prev = curr

if __name__ == "__main__":
    pd = multiprocessing.Process(target=main_deliver)
    pr = multiprocessing.Process(target=main_receiver)

    pr.start()
    time.sleep(4)

    pd.start()

    time.sleep(60)
    
