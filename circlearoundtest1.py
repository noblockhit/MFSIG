import time

step = 1
size = (20, 30)

for distance in range(max(size)//step):
    prnt = [["." for _ in range(size[1])] for _ in range(size[0])]
    for o in [(x+size[0]//2, distance+size[1]//2) for x in range(-distance, distance, 1)] + \
             [(distance+size[0]//2, y+size[1]//2) for y in range(distance, -distance, -1)] + \
             [(x+size[0]//2, -distance+size[1]//2) for x in range(distance, -distance, -1)] + \
             [(-distance+size[0]//2, y+size[1]//2) for y in range(-distance, distance, 1)]:
        if not (0 <= o[0] < size[0] and 0 <= o[1] < size[1]):
            continue
        prnt[o[0]][o[1]] = "o"
        out = ""
        for x in prnt:
            out += "".join(x) + "\n"
        print("\b" + out)
        time.sleep(.05)