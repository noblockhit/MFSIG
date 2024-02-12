from main import line_generator_gpu
from time import perf_counter_ns as pcns
import numpy as np
import matplotlib.pyplot as plt

def test(sizes):
    sizes = [1] + sizes ## PRERUN FOR INIT TIME
    times = []
    for size in sizes:
        print(size)
        begin = pcns()
        x = np.random.randint(10**3, size=size)
        y = np.random.randint(10**3, size=size)
        z = np.random.randint(10**3, size=size)
        out = [[], [], []]
        line_generator_gpu(x, y, z, [], out)
        times.append((pcns() - begin) * 10**-9)
        
    fig, ax = plt.subplots()
    ax.plot(sizes[1:], times[1:], linewidth=2.0)

    plt.show()
    
test([i for i in range(10**4, 90 * 10**4, 5*10**4)])