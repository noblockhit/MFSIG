from time import perf_counter_ns
import numpy
from PIL import Image


size = 20000
imarray = (numpy.random.rand(size,size,3) * 255).astype('uint8')

for idx, opt in enumerate([Image.NEAREST, Image.BOX, Image.BILINEAR, Image.HAMMING, Image.BICUBIC, Image.LANCZOS]):
    pilimg = Image.fromarray(imarray)
    start = perf_counter_ns()
    imout = pilimg.resize((size//10, size//10), opt)
    print(idx, opt, (perf_counter_ns() - start)*10**-9)