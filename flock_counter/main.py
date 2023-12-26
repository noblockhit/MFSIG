import multiprocessing as mp
from tkinter import filedialog
import time
import cv2
from rawloader import load_raw_image
from display3d import show as show3d
import pickle
import numpy as np
import collections
import pycuda.driver as drv
from pycuda.compiler import SourceModule


KEYING_BLUR_RADIUS = 1
KEYING_MASK_THRESHHOLD = 180
CANNY_THRESHHOLD_1 = 140
MIN_RADIUS = 25
MAX_RADIUS = 90
IMAGE_DISTANCE_TO_PIXEL_FACTOR = 20
MAXIMUM_PLANAR_DISTANCE_TO_SAME_TIP = 140
MAX_CORES_FOR_MP = 8
PREVIEW_IMAGE_HEIGHT = 900


colors = [
    (51, 104, 101),
    (20, 67, 37),
    (22, 117, 8),
    (16, 135, 0),
    (16, 2, 113),
    (57, 54, 6),
    (20, 80, 9),
    (103, 40, 25),
    (18, 7, 4),
    (50, 54, 117),
    (16, 4, 69),
    (34, 113, 137),
    (17, 24, 70),
    (102, 151, 147),
    (21, 16, 39),
    (145, 17, 48),
    (102, 35, 51),
    (51, 20, 39),
    (86, 0, 66),
    (56, 130, 102),
    (18, 2, 37),
    (20, 150, 96),
    (18, 18, 145),
    (22, 0, 69),
    (16, 36, 100),
    (17, 18, 87),
    (39, 131, 17),
    (103, 32, 121),
    (18, 73, 56),
    (128, 146, 80),
    (118, 21, 80)
]

class qs:
    name_and_pos = {}

    @staticmethod
    def show(name, data, alias=None):
        if alias is None:
            alias = name
        if alias in qs.name_and_pos.keys():
            qs.name_and_pos = {}
        if len(qs.name_and_pos) == 0:
            x = 30
        else:    
            x = list(qs.name_and_pos.values())[-1][0] + list(qs.name_and_pos.values())[-1][2]
        y = 30

        ar = data.shape[1] / data.shape[0]
        height = PREVIEW_IMAGE_HEIGHT
        width = int(ar*height)
        cv2.imshow(name, cv2.resize(data, (width, height)))
        cv2.moveWindow(name, x, y)
        qs.name_and_pos[alias] = (x, y, width, height)


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


imgs = collections.OrderedDict({})


def load_image(name):
    if any(name.lower().endswith(ending) for ending in FILE_EXTENTIONS["CV2"]):
        rgb = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)

    elif any(name.lower().endswith(ending) for ending in FILE_EXTENTIONS["RAW"]):
        rgb = load_raw_image(name, "auto")

    if rgb.shape[0] > rgb.shape[1]:
        rgb = cv2.rotate(rgb, cv2.ROTATE_90_CLOCKWISE)

    return rgb


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
    print(image_paths, len(image_paths))
    rgb_values = mp.Pool(min(MAX_CORES_FOR_MP, len(image_paths))).imap(load_image, image_paths)

    for idx, (name, rgb) in enumerate(zip(image_paths, rgb_values)):
        print("loaded", name)
        imgs[name] = rgb

    loading_time = (time.time_ns() - img_load_time_start) / (10 ** 9) 


def key_out_tips(img_in):
    blurred = cv2.medianBlur(img_in, KEYING_BLUR_RADIUS*2+1)
    mask = cv2.cvtColor(cv2.threshold(blurred, KEYING_MASK_THRESHHOLD, 1, cv2.THRESH_BINARY)[1], cv2.COLOR_RGB2GRAY)

    img = cv2.bitwise_and(img_in, img_in, mask=mask)
    edged = cv2.Canny(image=cv2.cvtColor(mask*255, cv2.COLOR_GRAY2BGR), threshold1=CANNY_THRESHHOLD_1, threshold2=255)
    return img, edged


def generate_circles(arg):
    try:
        temp_col_idx = 0
        idx, (image_name, input_img) = arg
        circles = []
        keyed, img = key_out_tips(input_img)
        # threshold
        thresh = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)[1]

        # get contours
        contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        hierarchy = contours[1] if len(contours) == 2 else contours[2]
        contours = contours[0] if len(contours) == 2 else contours[1]

        # get the actual inner list of hierarchy descriptions
        hierarchy = hierarchy[0]

        # count inner contours

        result = img.copy()
        result = cv2.merge([result,result,result])
        for cntr, hier in zip(contours, hierarchy):
            if hier[3] == -1:
                (x,y),radius = cv2.minEnclosingCircle(cntr)
                x, y, radius = int(x), int(y), int(radius)

                if MIN_RADIUS < radius < MAX_RADIUS:
                    cv2.drawContours(result, [cntr], 0, colors[temp_col_idx % len(colors)], 4)
                    temp_col_idx += 1
                    cv2.circle(result, (x, y), radius, (255,0,0), 2)
                    circles.append([image_name, x, y, radius, idx])
                            

        # qs.show(f"res {image_name[-15:]}", result, alias="Result from contours")
        return circles
    except TypeError:
        return [] 


def find_nearest_pow_2(val):
    for i in range(32):
        if val < 2**i:
            return 2**i


def line_generator_gpu(data_x, data_y, data_z, radiuses, lines):
    drv.init()
    dev = drv.Device(0)
    ctx = dev.make_context()

    mod = SourceModule("""
    __global__ void createLines(int *data_x, int *data_y, int *data_z,
                                int *length_arr, int* maximum_distance_arr,
                                int *lines_x, int *lines_y, int *lines_z) {
        const int length = length_arr[0];
        const int maximum_distance = maximum_distance_arr[0];

        const int thrd_i = blockIdx.x * blockDim.x + threadIdx.x;

        const int x = data_x[thrd_i];
        const int y = data_y[thrd_i];
        const int z = data_z[thrd_i];
        
        int distance = 2000000000;
                       

        for (int i = 0; i < length; i++) {
            int other_x = data_x[i];
            int other_y = data_y[i];
            int other_z = data_z[i];

            int height_distance = z - other_z;
            
            if (0 < height_distance && height_distance < 8) {
                int new_distance_2d = (x-other_x)*(x-other_x) + (y-other_y)*(y-other_y);
                int new_distance_3d = new_distance_2d + (z-other_z)*(z-other_z);
                if (new_distance_2d < maximum_distance && new_distance_3d < distance) {
                    distance = new_distance_3d;
                    lines_x[thrd_i*3+1] = other_x;
                    lines_y[thrd_i*3+1] = other_y;
                    lines_z[thrd_i*3+1] = other_z;
                }
            }
        }
        if (lines_x[thrd_i*3+1] > -1) {
            lines_x[thrd_i*3] = x;
            lines_y[thrd_i*3] = y;
            lines_z[thrd_i*3] = z;
        }
                       
    }""")

    create_lines = mod.get_function("createLines")

    MAX_THREADS = 1024

    grid_width = find_nearest_pow_2(len(data_x)/MAX_THREADS)
    lines_x = np.empty(len(data_x)*3, dtype=np.int32)
    lines_y = np.empty(len(data_y)*3, dtype=np.int32)
    lines_z = np.empty(len(data_z)*3, dtype=np.int32)

    lines_x[:] = -1
    lines_y[:] = -1
    lines_z[:] = -1
    
    try:
        create_lines(
            drv.In(np.array(data_x, dtype=np.int32)), drv.In(np.array(data_y, dtype=np.int32)), drv.In(np.array(data_z, dtype=np.int32)),
            drv.In(np.array([len(data_x)])), drv.In(np.array([MAXIMUM_PLANAR_DISTANCE_TO_SAME_TIP])), 
            drv.InOut(lines_x), drv.InOut(lines_y), drv.InOut(lines_z),
            block=(MAX_THREADS, 1, 1), grid=(grid_width, 1))
    except:
        ctx.pop()
        raise
    else:
        ctx.pop()

    lines[0] = [x if x >= 0 else None for x in lines_x]
    lines[1] = [y if y >= 0 else None for y in lines_y]
    lines[2] = [z if z >= 0 else None for z in lines_z]


def main():
    on_load_new_image()

    circles_list = mp.Pool(min(MAX_CORES_FOR_MP, len(imgs))).imap(generate_circles, list(enumerate(imgs.items())))

    circles = []
    for cs in circles_list:
        circles += cs
        if len(cs) > 0:
            print(f"evaluated {cs[0][0]}")

    plot_data_x, plot_data_y, plot_data_z, radiuses = zip(*[(center_x, center_y, image_idx, radius) for image_name, center_x, center_y, radius, image_idx in circles])


    lines = [
        [],
        [],
        []
    ]
    line_generator_gpu(plot_data_x, plot_data_y, plot_data_z, radiuses, lines)


    show3d((plot_data_x, plot_data_y, plot_data_z), lines)
    with open(b"tmp.3dgraph", "wb") as wf:
        pickle.dump({"circles":(plot_data_x, plot_data_y, plot_data_z), "lines": lines}, wf)
    input()


if __name__ == "__main__":
    main()
