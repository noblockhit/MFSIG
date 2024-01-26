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

    image_paths = sorted(image_paths)
    
    rgb_values = mp.Pool(len(image_paths)).imap(load_image, image_paths)

    for idx, (name, rgb) in enumerate(zip(image_paths, rgb_values)):
        imgs.append(cv2.fastNlMeansDenoisingColored(rgb ,None,10,10,7,21))

    loading_time = (time.time_ns() - img_load_time_start) / (10 ** 9)


def find_circles(img, minDist = 150, param1 = 100, param2 = 5, minRadius = 20, maxRadius = 70, blur_rad = 3):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, blur_rad) #cv2.bilateralFilter(gray,10,50,50)

    # docstring of HoughCircles: HoughCircles(image, method, dp, minDist[, circles[, param1[, param2[, minRadius[, maxRadius]]]]]) -> circles
    return cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)



def find_nearest_pow_2(val):
    for i in range(32):
        if val < 2**i:
            return 2**i


def render(radius, image):
    drv.init()
    dev = drv.Device(0)
    ctx = dev.make_context()

    mod = SourceModule("""
    __device__ int get_pos(int x, int y, int width, int color) {
        return (y * width + x) * 3 + color;
    }

    __global__ void compareAndPushSharpnesses(char *destination, double *sharpnesses, char *source, int *w_arr, int *h_arr,
                    int *r_arr) {
        int width = w_arr[0];
        int height = h_arr[0];
        int radius = r_arr[0];
        
        
        const int thrd_i = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (thrd_i > width*height) {
            return;
        }
        
        int center_x = thrd_i / height;
        int center_y = thrd_i % height;

        char center_b = source[get_pos(center_x, center_y, width, 0)];
        char center_g = source[get_pos(center_x, center_y, width, 1)];
        char center_r = source[get_pos(center_x, center_y, width, 2)];

        long delta = 0;

        int calculated_pixels = 0;

        
        for (int x = center_x-radius; x < center_x+radius+1; x++) {
            for (int y = center_y-radius; y < center_y+radius+1; y++) {
                if (x < 0 || y < 0 || x > width || y > height) {
                    continue;
                }
                
                if (x == center_x && y == center_y) {
                    continue;
                }
                
                float d = (float)(abs(abs(center_b) - abs(source[get_pos(x, y, width, 0)])) + abs(abs(center_g) - abs(source[get_pos(x, y, width, 1)])) + abs(abs(center_r) - abs(source[get_pos(x, y, width, 2)])));
                
                delta += (int)d;
                calculated_pixels++;
            }
        }
        double sharpness = (double)(delta) / (double)(calculated_pixels * 3 * 255);
        
        if (sharpness > sharpnesses[thrd_i]) {
            sharpnesses[thrd_i] = sharpness;
            destination[get_pos(center_x, center_y, width, 0)] = center_b;
            destination[get_pos(center_x, center_y, width, 1)] = center_g;
            destination[get_pos(center_x, center_y, width, 2)] = center_r;
        }
    }""")

    compareAndPushSharpnesses = mod.get_function("compareAndPushSharpnesses")

    MAX_THREADS = 1024
    
    width = int(image.shape[1])
    height = int(image.shape[0])

    composite_image_gpu = np.zeros((width * height * 3), dtype=np.uint8)
    sharpnesses_gpu = np.zeros((width * height), dtype=float)

    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    total_pixels = width*height

    grid_width = find_nearest_pow_2(total_pixels/MAX_THREADS)
    
    try:
        compareAndPushSharpnesses(
            drv.InOut(composite_image_gpu), drv.InOut(sharpnesses_gpu), drv.In(bgr.flatten(order="K")),
            drv.In(np.array([width])), drv.In(np.array([height])), drv.In(np.array([radius])),
            block=(MAX_THREADS, 1, 1), grid=(grid_width, 1))
    except:
        ctx.pop()
        raise
        
    ctx.pop()
    return width, height, composite_image_gpu, sharpnesses_gpu

def convert_color_arr_to_image(arr, width, height):
    return cv2.cvtColor(arr.reshape(height, width, 3), cv2.COLOR_BGR2RGB)

def convert_gray_arr_to_image(arr, width, height):
    return cv2.cvtColor(cv2.flip(cv2.rotate(arr.reshape(width, height), cv2.ROTATE_90_CLOCKWISE), 1), cv2.COLOR_GRAY2RGB)



def show(name, data):
    ar = data.shape[1] / data.shape[0]
    height = 900
    width = int(ar*height)
    print(width, height)
    cv2.imshow(name, cv2.resize(data, (width, height)))
    cv2.moveWindow(name, 50, 50)

if __name__ == "__main__":
    on_load_new_image()
    print("rendering:")
    outputs = [(img, *render(10, img)) for img in imgs]
    thresh = 80

    for img, width, height, composite_image_gpu, sharpnesses_gpu in outputs:
        output_img = convert_color_arr_to_image(composite_image_gpu, width, height)
        output_sharpness = convert_gray_arr_to_image(cv2.normalize(sharpnesses_gpu, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U), width, height)

        sharpness_bw = cv2.threshold(output_sharpness, thresh, 255, cv2.THRESH_BINARY)[1]

        tips = find_circles(sharpness_bw)
        tips = np.uint16(np.around(tips))[0]
        for i in tips:
            cv2.circle(sharpness_bw, (i[0], i[1]), i[2], (0, 255, 0), 3)

        show("img", img)
        show("sharpness map", output_sharpness)
        show("sharpness bw", sharpness_bw)

        cv2.waitKey(0)