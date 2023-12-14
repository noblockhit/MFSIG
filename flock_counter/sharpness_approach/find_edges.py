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


    ## denoising
    return cv2.medianBlur(rgb, 5)


imgs = []

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
        imgs.append(rgb)

    loading_time = (time.time_ns() - img_load_time_start) / (10 ** 9)

if __name__ == "__main__":
    on_load_new_image()

    minDist = 120
    param1 = 50 #500
    param2 = 40 #200 #smaller value-> more false circles
    minRadius = 4
    maxRadius = 150 #10
    for img in imgs:
        edges = cv2.Canny(image=img, threshold1=100, threshold2=200)
        
        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
        print(circles)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0,:]:
                cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)


        cv2.imshow("img", img)
        cv2.imshow("edges", edges)
        cv2.waitKey(1000) 