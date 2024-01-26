import rawpy
import cv2
import rawpy
import multiprocessing as mp
from tkinter import filedialog
import time
import cv2
import numpy as np
from typing import Union


def load_raw_image(path, brightness: Union[float, int, str]=1):
    with rawpy.imread(path) as raw:
        # Access the raw image data
        raw_data = raw.raw_image
        height, width = raw_data.shape
        raw_array = raw_data.reshape((height, width, 1))

        # Demosaic the raw array
        try:
            brightness = float(brightness)
        except ValueError:
            if brightness == "auto":
                uint_16_demosaiced = cv2.cvtColor(raw_array, 46)
                brightness = 65535/np.amax(uint_16_demosaiced)
                return cv2.convertScaleAbs(uint_16_demosaiced, alpha=(255.0/65535.0) * 16 * brightness)
            raise ValueError(f"brightness must be \"auto\", float or an int, not {brightness}")
        else:
            return cv2.convertScaleAbs(cv2.cvtColor(raw_array, 46), alpha=(255.0/65535.0) * 16 * brightness)


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
        
        
        
