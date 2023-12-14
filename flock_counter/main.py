import tkinter as tk
import customtkinter
from PIL import ImageTk, Image
import rawpy
from tkinter import filedialog, messagebox, RIGHT, LEFT
import time
import gc
import os
import cv2
import numpy as np
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import copy
from math import sqrt
from threading import Thread
import multiprocessing as mp
import subprocess
import sys
import statistics
import matplotlib.pyplot as plt
from pathlib import Path
import tkinter as tk
import customtkinter
from PIL import ImageTk, Image
import rawpy
from tkinter import filedialog, messagebox, RIGHT, LEFT
import time
import gc
import os
import cv2
import numpy as np
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import copy
from math import sqrt
from threading import Thread
import multiprocessing as mp
import subprocess
import sys
import statistics
import math


def show(name, data, x, y):
    ar = data.shape[1] / data.shape[0]
    height = 900
    width = int(ar*height)
    print(width, height)
    cv2.imshow(name, cv2.resize(data, (width, height)))
    cv2.moveWindow(name, x, y)
    
FILE_EXTENTIONS = {
    "RAW": [
        ".nef",
        ".arw",
        ".raw",
    ],
    "CV2": [
        ".jpeg",
        ".jpg",
        ".png",
        ".tiff",
        ".tif"
    ],
}

def load_image(name):
    if any(name.lower().endswith(ending) for ending in FILE_EXTENTIONS["CV2"]):
        rgb = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)

    elif any(name.lower().endswith(ending) for ending in FILE_EXTENTIONS["RAW"]):
        with rawpy.imread(name) as raw:
            rgb = raw.postprocess(use_camera_wb=False)

    if rgb.shape[0] > rgb.shape[1]:
        rgb = cv2.rotate(rgb, cv2.ROTATE_90_CLOCKWISE)

    return rgb

imgs = {}

def on_load_new_image():
    global loading_time
    selected_img_files = filedialog.askopenfiles(title="Open Images for the render queue", filetypes=[("Image-files", ".tiff .tif .png .jpg .jpeg .RAW .NEF")])
    if not selected_img_files:
        return

    img_load_time_start = time.time_ns()
    image_paths = []

    for f in selected_img_files:
        image_paths.append(f.name)
    
    rgb_values = mp.Pool(len(image_paths)).imap(load_image, image_paths)

    for idx, (name, rgb) in enumerate(zip(image_paths, rgb_values)):
        imgs[name] = rgb

    loading_time = (time.time_ns() - img_load_time_start) / (10 ** 9) 


def key_out_tips(img_in):
    blurred = cv2.medianBlur(img_in, 7)
    mask = cv2.cvtColor(cv2.threshold(blurred, 155, 1, cv2.THRESH_BINARY)[1], cv2.COLOR_RGB2GRAY)

    img = cv2.bitwise_and(img_in, img_in, mask=mask)
    edged = cv2.Canny(image=img, threshold1=120, threshold2=200)
    return edged


def main():
    on_load_new_image()
    
    print("done")

    for input_img in imgs.values():
        img = key_out_tips(input_img)
        # threshold
        thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)[1]

        # get contours
        contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        hierarchy = contours[1] if len(contours) == 2 else contours[2]
        contours = contours[0] if len(contours) == 2 else contours[1]

        # get the actual inner list of hierarchy descriptions
        hierarchy = hierarchy[0]

        # count inner contours
        result = img.copy()
        result = cv2.merge([result,result,result])
        for cntr, hier in zip(contours, hierarchy):
            # discard outermost no parent contours and keep innermost no child contours
            # hier = indices for next, previous, child, parent
            # no parent or no child indicated by negative values
            if hier[3] == -1:
                try:
                    M = cv2.moments(cntr)
                    center_x = int(M["m10"] / M["m00"])
                    center_y = int(M["m01"] / M["m00"])
                    x,y,w,h = cv2.boundingRect(cntr)
                except ZeroDivisionError:
                    continue
                else:
                    if w > 25 < h:
                        if w < 130 > h:
                            cv2.drawContours(result, [cntr], 0, (0,0,255), 2)
                            cv2.rectangle(input_img, (x, y), (x + w, y + h), (0,255,255), 1)
                            cv2.circle(result, (center_x, center_y), 2, (255,0,255), 2)

        

        # show result
        cv2.destroyAllWindows()
        show("img", input_img, 0, 30)
        show("result", result, 900, 30)
        cv2.waitKey(300)

if __name__ == "__main__":
    main()
