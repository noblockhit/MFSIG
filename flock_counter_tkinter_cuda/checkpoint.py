import time


class Checkpoint:
    time = time.perf_counter_ns()
    def __init__(self, *args, **kwds):
        delta = time.perf_counter_ns() - Checkpoint.time
        Checkpoint.time = time.perf_counter_ns()
        if len(args) == 0 and len(kwds) == 0:
            return
        print(f"{delta*10**-9:.5f} seconds for {' '.join(args)} {str(kwds)[1:-1]}")