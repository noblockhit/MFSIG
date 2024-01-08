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
MAX_CORES_FOR_MP = mp.cpu_count()-1
PREVIEW_IMAGE_HEIGHT = 1000


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
        if ar > 1:
            data = cv2.rotate(data, cv2.ROTATE_90_CLOCKWISE)
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



def key_out_tips(img_in, keying_blur_radius, keying_mask_threshhold, canny_threshhold_1):
    blurred = cv2.medianBlur(img_in, keying_blur_radius*2+1)
    mask = cv2.cvtColor(cv2.threshold(blurred, keying_mask_threshhold, 1, cv2.THRESH_BINARY)[1], cv2.COLOR_RGB2GRAY)

    keyed = cv2.resize(cv2.bitwise_and(img_in, img_in, mask=mask), (img_in.shape[1]//3, img_in.shape[0]//3)) 
    edges = cv2.Canny(image=cv2.cvtColor(mask*255, cv2.COLOR_GRAY2BGR), threshold1=canny_threshhold_1, threshold2=255)
    return keyed, edges


def generate_circles(idx, image_name, input_img, keying_blur_radius, keying_mask_threshhold, canny_threshhold_1, min_radius, max_radius):
    try:
        temp_col_idx = 0
        circles = []
        keyed, edges = key_out_tips(input_img, keying_blur_radius, keying_mask_threshhold, canny_threshhold_1)
        # threshold
        thresh = cv2.threshold(edges, 1, 255, cv2.THRESH_BINARY)[1]

        # get contours
        contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        del thresh ## OPTIMIZATION
        
        hierarchy = contours[1] if len(contours) == 2 else contours[2]
        contours = contours[0] if len(contours) == 2 else contours[1]

        hierarchy = hierarchy[0]

        # count inner contours

        result = cv2.merge([edges,edges,edges])
        del edges ## OPTIMIZATION
        for cntr, hier in zip(contours, hierarchy):
            if hier[3] != -1:
                continue
            (x,y),radius = cv2.minEnclosingCircle(cntr)
            x, y, radius = int(x), int(y), int(radius)

            if min_radius < radius < max_radius:
                cv2.drawContours(result, [cntr], 0, colors[temp_col_idx % len(colors)], 4)
                temp_col_idx += 1
                cv2.circle(result, (x, y), radius, (255,0,0), 2)
                circles.append([image_name, x, y, radius, idx])
        result = cv2.resize(result, (keyed.shape[1], keyed.shape[0]))
        return keyed, result, circles
    except TypeError:
        return [] 


def load_and_evaluate_image(args):
    idx, imagename_or_img, keying_blur_radius, keying_mask_threshhold, canny_threshhold_1, min_radius, max_radius = args
    if isinstance(imagename_or_img, str):
        name = imagename_or_img
        if any(name.lower().endswith(ending) for ending in FILE_EXTENTIONS["CV2"]):
            image = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)

        elif any(name.lower().endswith(ending) for ending in FILE_EXTENTIONS["RAW"]):
            image = load_raw_image(name, 35)

        if image.shape[0] > image.shape[1]:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    elif isinstance(imagename_or_img, tuple):
        name, image = imagename_or_img
    else:
        raise ValueError("Second item in `args` is neither a tuple of a name and an image nor just the path to a file")

    keyed, edges_result, circles = generate_circles(idx, name, image, keying_blur_radius, keying_mask_threshhold, canny_threshhold_1, min_radius, max_radius)
    return idx, name, image, keyed, edges_result, circles


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
                                int *length_arr, int* maximum_distance_arr, int*image_distance_to_pixel_factor_arr,
                                int *lines_x, int *lines_y, int *lines_z) {
        const int length = length_arr[0];
        const int maximum_distance = maximum_distance_arr[0];
        const int image_distance_to_pixel_factor = image_distance_to_pixel_factor_arr[0];

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
                int new_distance_3d = new_distance_2d + (z-other_z)*(z-other_z)*image_distance_to_pixel_factor*image_distance_to_pixel_factor;
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
            drv.In(np.array([len(data_x)])), drv.In(np.array([MAXIMUM_PLANAR_DISTANCE_TO_SAME_TIP])), drv.In(np.array([IMAGE_DISTANCE_TO_PIXEL_FACTOR])),
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


def on_load_new_image():
    imgs = collections.OrderedDict({})

    selected_img_files = filedialog.askopenfiles(title="Open Images for the render queue", filetypes=[("Image-files", ".tiff .tif .png .jpg .jpeg .RAW .NEF")])
    if not selected_img_files:
        return

    image_paths = []

    for f in selected_img_files:
        image_paths.append(f.name)
    image_paths = sorted(image_paths)
    
    satisfied = False
    first_load = True
    shown_index = 0
    while not satisfied:
        all_circles = []
        if first_load:
            args = enumerate(image_paths)
            first_load = False
        else:
            args = enumerate(imgs.items())
        args = [(idx, img, KEYING_BLUR_RADIUS, KEYING_MASK_THRESHHOLD, CANNY_THRESHHOLD_1, MIN_RADIUS, MAX_RADIUS) for idx, img in args]
        images_and_circles = mp.Pool(min(MAX_CORES_FOR_MP, len(image_paths))).map(load_and_evaluate_image, args)
        

        for _, name, image, _, _, circles in images_and_circles:
            if name not in imgs.keys():
                imgs[name] = image
            all_circles += circles
        
        while True:
            idx, name, image, keyed, edges_result, _ = images_and_circles[shown_index]
            # cv2.destroyAllWindows()
            print(shown_index)
            qs.show(f"image", image, alias="image")
            qs.show(f"keyed", keyed, alias="keyed")
            qs.show(f"edges_result", edges_result, alias="edges_result")
            key_ord = cv2.waitKey(0)
            
            if key_ord == ord("q"):
                shown_index -= 1
            elif key_ord == ord("e"):
                shown_index += 1
            elif key_ord == ord("r"):
                cv2.destroyAllWindows()
                break
            
            elif key_ord == ord("s"):
                satisfied = True
                cv2.destroyAllWindows()
                break
            
            shown_index = (shown_index+len(images_and_circles)) % len(images_and_circles)

        
    return imgs, all_circles


def main():
    imgs, circles = on_load_new_image()
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
