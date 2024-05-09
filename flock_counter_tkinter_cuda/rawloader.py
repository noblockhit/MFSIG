import rawpy
import cv2
import rawpy
import multiprocess as mp
from tkinter import filedialog
from time import perf_counter_ns as pcns
import cv2
import numpy as np
from typing import Union
from matplotlib import pyplot as plt



def get_x_y_histos(img):
    x_hist = np.zeros(img.shape[1])
    for x in range(img.shape[1]):
        x_hist[x] = np.sum(img[:, x])
    
    y_hist = np.zeros(img.shape[0])
    for y in range(img.shape[0]):
        y_hist[y] = np.sum(img[y, :])
        
    return x_hist, y_hist
    
    
def _load_raw_from_file(path):
    print(path)
    with rawpy.imread(path) as raw:
        # Access the raw image data
        raw_data = raw.raw_image
        height, width = raw_data.shape
        raw_array = raw_data.reshape((height, width, 1))
        return cv2.cvtColor(raw_array, 46)
    
def worker(path):
    demos = _load_raw_from_file(path)
    flattened_data = cv2.resize(demos, (demos.shape[0]//4, demos.shape[1]//4)).flatten()
    sorted_data = np.sort(flattened_data)
    upper_quarter_index = int(len(sorted_data) * 0.95)

    upper_quarter = sorted_data[upper_quarter_index:]

    median_upper_quarter = np.median(upper_quarter)
    return demos, (65535/median_upper_quarter/8)
    
def mp_load_and_getbrightness(paths):
    with mp.Pool(mp.cpu_count()//2) as pool:
        return zip(*pool.map(worker, paths))
    

    
def load_raw_images(paths, brightness: Union[float, int, str]=1):
    if brightness == "auto":
        demosaiced_images, brightnesses = mp_load_and_getbrightness(paths)
        brightness = sum(brightnesses)/len(brightnesses)
    else:
        demosaiced_images = [_load_raw_from_file(path) for path in paths]
    print(demosaiced_images)
    imgs = [cv2.convertScaleAbs(demos, alpha=float(brightness/257)) for demos in demosaiced_images]
    
    with mp.Pool(mp.cpu_count()) as pool:
        x_y_histos = pool.map(get_x_y_histos, imgs)
    
    return imgs, x_y_histos


if __name__ == "__main__":
    class qs:
        name_and_pos = {}

        @staticmethod
        def show(name, data):
            if name in qs.name_and_pos.keys():
                qs.name_and_pos = {}
            if len(qs.name_and_pos) == 0:
                x = 30
            else:    
                x = list(qs.name_and_pos.values())[-1][0] + list(qs.name_and_pos.values())[-1][2]
            y = 30

            ar = data.shape[1] / data.shape[0]
            height = 1100
            width = int(ar*height)
            cv2.imshow(name, cv2.resize(data, (width, height)))
            cv2.moveWindow(name, x, y)
            qs.name_and_pos[name] = (x, y, width, height)


    selected_img_files = filedialog.askopenfiles(title="Open Images for the render queue", filetypes=[("Image-files", ".tiff .tif .png .jpg .jpeg .RAW .NEF")])

    image_paths = []

    for f in selected_img_files:
        image_paths.append(f.name)
    image_paths = sorted(image_paths)
    print(image_paths)

    for idx, name in enumerate(image_paths):
        rgb = load_raw_image(name, brightness=1)
        print("loaded", name)
        qs.show(name, rgb)
    cv2.waitKey(0)

    
    for idx, name in enumerate(image_paths):
        rgb = load_raw_image(name, brightness="auto")
        print("loaded", name)
        qs.show(name, rgb)
    cv2.waitKey(0)
        
        
        
