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
from image_array_converter import convert_color_arr_to_image, convert_gray_arr_to_image
from math import sqrt
import colorama
colorama.init()

customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("dark-blue")

root = customtkinter.CTk()
root.geometry("800x800")
root.resizable(height=800, width=800)

image_arr_dict = {}

image_preview_frame = VerticalScrolledFrame(root)

def add_image_to_scrollbar(img, name):
    container = customtkinter.CTkFrame(image_preview_frame.interior)
    container.pack(padx=0, pady=5)

    target_diag_size = sqrt(2)*150

    curr_diag_size = sqrt(img.size[0]**2+img.size[1]**2)

    scaling_factor = curr_diag_size / target_diag_size
    print(sqrt((img.size[0]//scaling_factor)**2 + (img.size[1]//scaling_factor)**2))
    tk_img = customtkinter.CTkImage(img, size=(img.size[0]//scaling_factor, img.size[1]//scaling_factor))
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

FILE_EXTENTIONS = {
    "RAW": [
        ".nef",
        ".arw",
        ".tiff",
        ".raw",
    ],
    "CV2": [
        ".jpeg",
        ".jpg",
        ".png",
    ],
}

def load_new_image():
    selected_img_files = filedialog.askopenfiles(filetypes=[("Image-files", ".png .jpg .jpeg .RAW .NEF")])
    for idx, f in enumerate(selected_img_files):
        if os.path.basename(f.name) in image_arr_dict.keys():
            continue

        print(f"Opening image {os.path.basename(f.name)}, {idx+1} out of {len(selected_img_files)}")
        if any(f.name.lower().endswith(ending) for ending in FILE_EXTENTIONS["CV2"]):
            bgr = cv2.imread(f.name)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        elif any(f.name.lower().endswith(ending) for ending in FILE_EXTENTIONS["RAW"]):
            with rawpy.imread(f.name) as raw:
                rgb = raw.postprocess(use_camera_wb=True)
        img = Image.fromarray(rgb)

        add_image_to_scrollbar(img, os.path.basename(f.name))

        image_arr_dict[os.path.basename(f.name)] = rgb

        del img
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


global img_panel
global changes_panel
global img_panel_packed
global changes_panel_packed
img_panel = None
changes_panel = None
img_panel_packed = False
changes_panel_packed = False


def pack_img_panel():
    global img_panel
    global img_panel_packed
    if img_panel:
        img_panel.pack(padx=5, pady=5, expand=True, fill = "both")
        img_panel_packed = True


def pack_changes_panel():
    global changes_panel
    global changes_panel_packed
    if changes_panel:
        changes_panel.pack(padx=5, pady=5, expand=True, fill = "both")
        changes_panel_packed = True


def unpack_img_panel():
    global img_panel
    global img_panel_packed
    img_panel_packed = False
    if img_panel:
        img_panel.pack_forget()
        return 1
    return 0


def unpack_changes_panel():
    global changes_panel
    global changes_panel_packed
    changes_panel_packed = False
    if changes_panel:
        changes_panel.pack_forget()
        return 1
    return 0


def render():
    global img_panel
    global changes_panel
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
    previous_sharpnesses = [np.zeros((width * height), dtype=float)]
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
        changed_indecies = np.argwhere(previous_sharpnesses[i+1] - previous_sharpnesses[i] > 0)
        
        np.put(changes_arr, changed_indecies, [i+1])

    ## statistics
    statistics_calc_time_start = time.time()
    for number, count in dict(zip(*np.unique(changes_arr, return_counts=True))).items():
        if number == 0:
            continue
        print(f"Pixels used from image {list(image_arr_dict.keys())[number-1]} {count} ({(count / len(changes_arr) * 100):.2f}%)")
    
    print(f"Statistics took {time.time() - statistics_calc_time_start} seconds to compute")
        

    changes_arr = changes_arr * int(255 / len(image_arr_dict))

    unpack_img_panel()
    unpack_changes_panel()

    changes_img = convert_gray_arr_to_image(changes_arr, width, height)
    changes_panel = PreviewImage(processing_frame, image = changes_img)
    on_show_changes_checkbox()

    processed_img = convert_color_arr_to_image(composite_image_gpu, width, height)
    img_panel = PreviewImage(processing_frame, image = processed_img)
    on_show_output_checkbox()


image_preview_frame.pack(ipadx=20, fill = "y", side=RIGHT)
button = customtkinter.CTkButton(master=image_preview_frame.interior, text="Load new image", command=load_new_image)
button.pack(pady=(12, 5))


processing_frame = customtkinter.CTkFrame(root)
processing_frame.pack(expand = True, padx=0, pady=0, side=LEFT, fill = "both")

settings_frame = customtkinter.CTkFrame(root, width=100)
settings_frame.pack(padx=10, pady=0, fill = "y")


def on_show_changes_checkbox():
    if show_changes_intvar.get() == 1:
        pack_changes_panel()
    else:
        unpack_changes_panel()

show_changes_intvar = customtkinter.IntVar(value=1)
show_changes_checkbox = customtkinter.CTkCheckBox(settings_frame, text="Show Changes Image", variable=show_changes_intvar,
                                                 onvalue=1, offvalue=0, command=on_show_changes_checkbox)
show_changes_checkbox.pack(pady = 10, padx=5)

def on_show_output_checkbox():
    if show_output_intvar.get() == 1:
        pack_img_panel()
    else:
        unpack_img_panel()

show_output_intvar = customtkinter.IntVar(value=1)
show_output_checkbox = customtkinter.CTkCheckBox(settings_frame, text="Show Output Image", variable=show_output_intvar,
                                                 onvalue=1, offvalue=0, command=on_show_output_checkbox)
show_output_checkbox.pack(pady = 10, padx=5)




render_button = customtkinter.CTkButton(processing_frame, text="Render opened images", command=render)
render_button.pack(pady=(12, 5))

root.mainloop()
