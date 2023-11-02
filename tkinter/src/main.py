import tkinter as tk
import customtkinter
from scrollable_frame import VerticalScrolledFrame
from growing_image import GrowingImage
from preview_image import PreviewImage
from PIL import ImageTk, Image
import rawpy
from tkinter import filedialog, messagebox, RIGHT, LEFT
import time
import gc
import os
import cv2
import numpy as np
import pycuda.autoinit ## DONT REMOVE THIS
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import copy

customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("dark-blue")

root = customtkinter.CTk()
root.geometry("600x600")
root.resizable(height=500, width=500)

image_arr_dict = {}

image_preview_frame = VerticalScrolledFrame(root)

def add_image_to_scrollbar(img, name):
    container = customtkinter.CTkFrame(image_preview_frame.interior)
    container.pack(padx=0, pady=5)

    tk_img = customtkinter.CTkImage(img, size=(img.size[0]//20, img.size[1]//20))
    img_panel = customtkinter.CTkLabel(container, image = tk_img, text="")
    img_panel.pack(padx=5, pady=5)
    
    name_panel = customtkinter.CTkLabel(container, text=name)
    name_panel.pack(padx=2, pady=2)


    def destroy_container():
        img_panel.pack_forget()
        destroy_button.pack_forget()
        container.pack_forget()
        name_panel.pack_forget()

        img_panel.destroy()
        destroy_button.destroy()
        container.destroy()
        name_panel.destroy()

        image_arr_dict.pop(name)

        gc.collect()

    destroy_button = customtkinter.CTkButton(container, text="Remove Image", command=destroy_container)
    destroy_button.pack(padx=2, pady=5)

def load_new_image():
    selected_img_files = filedialog.askopenfiles(filetypes=[("Image-files", ".png .jpg .jpeg .RAW .NEF")])
    for idx, f in enumerate(selected_img_files):
        if os.path.basename(f.name) in image_arr_dict.keys():
            continue

        print(f"Opening image {os.path.basename(f.name)}, {idx+1} out of {len(selected_img_files)}")
        raw = rawpy.imread(f.name)
        rgb = raw.postprocess(use_camera_wb=True)
        img = Image.fromarray(rgb)
        add_image_to_scrollbar(img, os.path.basename(f.name))

        image_arr_dict[os.path.basename(f.name)] = rgb

        del img, raw
        gc.collect()



def find_nearest_pow_2(val):
    for i in range(32):
        if val < 2**i:
            return 2**i

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


def render():
    img1 = list(image_arr_dict.values())[0]
    RESIZE = 100
    MAX_THREADS = 1024
    radius = 10
    if RESIZE != 100:
        scale_percent = RESIZE  # percent of original size
        width = int(img1.shape[1] * scale_percent / 100)
        height = int(img1.shape[0] * scale_percent / 100)
        dim = (width, height)
        img1 = cv2.resize(img1, dim, interpolation=cv2.INTER_AREA)
    else:
        width = int(img1.shape[1])
        height = int(img1.shape[0])

    composite_image_gpu = np.zeros((width * height * 3), dtype=np.uint8)
    sharpnesses_gpu = np.zeros((width * height), dtype=float)
    previous_sharpnesses = [sharpnesses_gpu]
    changes_arr = np.zeros((width * height), dtype=np.uint8)

    for i, (name, rgb) in enumerate(image_arr_dict.items()):
        print("rendering:", name, "at index of", i)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        total_pixels = width*height

        grid_width = find_nearest_pow_2(total_pixels/MAX_THREADS)
        
        previous_sharpnesses.append(np.zeros((width * height), dtype=np.uint32))

        compareAndPushSharpnesses(
            drv.InOut(composite_image_gpu), drv.InOut(sharpnesses_gpu), drv.In(bgr.flatten(order="K")),
            drv.In(np.array([width])), drv.In(np.array([height])), drv.In(np.array([radius])),
            block=(MAX_THREADS, 1, 1), grid=(grid_width, 1))
        
        previous_sharpnesses[i+1] = copy.deepcopy(sharpnesses_gpu)
        np.put(changes_arr, np.equal(previous_sharpnesses[i],
                                     previous_sharpnesses[i+1]).nonzero()[0], [i])
        
    changes_arr = changes_arr * int(256 / len(image_arr_dict))
    scale_percent_prog = 15  # percent of original size
    width_prog = int(bgr.shape[1] * scale_percent_prog / 100)
    height_prog = int(bgr.shape[0] * scale_percent_prog / 100)
    changes_array_2d = cv2.flip(cv2.resize(cv2.rotate(changes_arr.reshape(width, height), cv2.ROTATE_90_CLOCKWISE), (width_prog, height_prog)), 1)

    
    processed_img = cv2.cvtColor(composite_image_gpu.reshape(height, width, 3), cv2.COLOR_BGR2RGB)   
    img_panel = PreviewImage(processing_frame, image = processed_img)
    img_panel.pack(padx=5, pady=5, expand=True, fill = "both")

    changes_img = cv2.cvtColor(changes_array_2d, cv2.COLOR_GRAY2RGB)
    changes_panel = PreviewImage(processing_frame, image = changes_img)
    changes_panel.pack(padx=5, pady=5, expand=True, fill = "both")




image_preview_frame.pack(expand = True, fill = "both", side=RIGHT)
button = customtkinter.CTkButton(master=image_preview_frame.interior, text="Load new image", command=load_new_image)
button.pack(pady=(12, 5))

processing_frame = customtkinter.CTkFrame(root)
processing_frame.pack(expand = True, padx=0, pady=0, side=LEFT, fill = "both")

render_button = customtkinter.CTkButton(processing_frame, text="Render opened images", command=render)
render_button.pack(pady=(12, 5))

root.mainloop()
