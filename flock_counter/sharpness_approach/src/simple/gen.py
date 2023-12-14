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

    global spot_selected
    spot_selected = None

    def select_spot(event,x,y,flags,param):
        global spot_selected
        if event == cv2.EVENT_LBUTTONDBLCLK:
            spot_selected = x, y

    while True:
        for img in imgs:
            cv2.imshow("select spot", img)
            cv2.moveWindow("select spot", 50, 50)
            cv2.waitKey(100)
            cv2.setMouseCallback("select spot", select_spot)

        if spot_selected is not None:
            break

    print(spot_selected)
    x, y = spot_selected
    width = height = 800
    global dir

    def save_image(name, cv2_image):
        global dir
        cv2.imwrite(str(Path(dir) / Path(name)), cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))
    
    def save():
        global dir
        dir = filedialog.askdirectory(title="Save as filenames")

    save()

    for idx, img in enumerate(imgs):
        print("cropped: ")
        c = img[y-height//2:y+height//2, x-width//2:x+width//2]
        save_image(f"{idx}.tiff", c)

        cv2.imshow("cropped", c)
        cv2.waitKey(1000)
    