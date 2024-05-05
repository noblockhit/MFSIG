import rawpy
import cv2
import rawpy
import multiprocessing as mp
from tkinter import filedialog
from time import perf_counter_ns as pcns
import cv2
import numpy as np
from typing import Union
from matplotlib import pyplot as plt

global prev_brightness
prev_brightness = None

def _load_raw_from_file(path):
    print(path)
    with rawpy.imread(path) as raw:
        # Access the raw image data
        raw_data = raw.raw_image
        height, width = raw_data.shape
        raw_array = raw_data.reshape((height, width, 1))
        return cv2.cvtColor(raw_array, 46)
    

def load_raw_images(paths, brightness: Union[float, int, str]=1):
    demosaiced_images = [_load_raw_from_file(path) for path in paths]
    if brightness == "auto":
        brightnesses = []
        for demos in demosaiced_images:
            print(demos.shape)
            flattentime = pcns()
            flattened_data = cv2.resize(demos, (demos.shape[0]//4, demos.shape[1]//4)).flatten()
            print(f"flatten time: {(pcns()-flattentime)*10**-9:.5f}")
            
            # Sort the flattened array
            sortedtime = pcns()
            sorted_data = np.sort(flattened_data)
            print(f"sort time: {(pcns()-sortedtime)*10**-9:.5f}")

            # Calculate the index of the median of the upper quarter
            
            upper_quarter_index = int(len(sorted_data) * 0.95)

            # Get the upper quarter of the sorted array
            upper_quarter = sorted_data[upper_quarter_index:]

            # Calculate the median of the upper quarter
            mediantime = pcns()
            median_upper_quarter = np.median(upper_quarter)
            print(f"median time: {(pcns()-mediantime)*10**-9:.5f}")
            brightnesses.append(65535/median_upper_quarter/8)
        brightness = sum(brightnesses)/len(brightnesses)
    # else:
    #     raise ValueError(f"brightness must be \"auto\", float or an int, not {brightness}")
    imgs = [cv2.convertScaleAbs(demos, alpha=float((255.0/65535.0) * brightness)) for demos in demosaiced_images]
    return imgs
    
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
        
        
        
