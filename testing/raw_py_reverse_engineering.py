import rawpy
import cv2
import numpy as np
import time
from functools import reduce


with rawpy.imread(r"D:\images\MFSIG\Insekt1\src\DSC_0001.NEF") as raw:
    data = raw.raw_image

    
    start = time.time()
    # norm = cv2.normalize(data, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)
    # norm = cv2.normalize(cv2.cvtColor(data, cv2.COLOR_BAYER_BG2BGR), dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)
    norm = cv2.cvtColor(cv2.normalize(data, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX), cv2.COLOR_BayerBG2BGR)
    norm = (norm/256).astype("uint8")
    scale_percent = 20  # percent of original size
    width = int(norm.shape[1] * scale_percent / 100)
    height = int(norm.shape[0] * scale_percent / 100)

    from_data = cv2.resize(norm, (width, height))
    print(f"Extracting took {time.time() - start} seconds")

    start = time.time()
    from_process = cv2.cvtColor(cv2.resize(cv2.rotate(raw.postprocess(), cv2.ROTATE_90_CLOCKWISE), (width, height)), cv2.COLOR_RGB2BGR)
    print(f"Processing took {time.time() - start} seconds")

    print(list(map(list, from_data[0][0:5])))
    print(list(map(list, from_process[0][0:5])))

    cv2.imshow("img1", from_data)
    cv2.imshow("pst", from_process)
    cv2.waitKey(0)