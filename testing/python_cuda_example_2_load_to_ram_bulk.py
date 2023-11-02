import rawpy
import cv2
import os
import copy
import numpy as np
import time
import pycuda.autoinit ## DONT REMOVE THIS
import pycuda.driver as drv
from pycuda.compiler import SourceModule
from pathlib import Path
from sbNative.runtimetools import get_path


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


def nparr_to_arr(a, depth=3):
    if depth == 3:
        return [["".join([hex(z)[-2:] for z in y]) for y in x] for x in a]

    if depth == 2:
        return [[float(y) for y in x] for x in a]


def get_sharpness(image, center_y, center_x, radius):
    center_b = int(image[center_x, center_y][0])
    center_g = int(image[center_x, center_y][1])
    center_r = int(image[center_x, center_y][2])

    w, h, *_ = image.shape

    delta_b = 0
    delta_g = 0
    delta_r = 0

    calculated_pixels = 0

    for x in range(center_x-radius, center_x+radius+1):
        for y in range(center_y-radius, center_y+radius+1):
            if x < 0 or y < 0 or x + 1 > w or y + 1 > h:
                continue

            if x == center_x and y == center_y:
                continue

            b = int(image[x, y][0])
            g = int(image[x, y][1])
            r = int(image[x, y][2])

            delta_b += abs(center_b - b)
            delta_g += abs(center_g - g)
            delta_r += abs(center_r - r)

            calculated_pixels += 1

    return (delta_b + delta_g + delta_r) / (calculated_pixels * 3 * 255)


def find_nearest_pow_2(val):
    for i in range(32):
        if val < 2**i:
            return 2**i


def main():
    with open(str(get_path() / "input_paths.txt")) as rf:
        path = Path(rf.read())

    sharpnesses = None
    composite_image = None
    sharpnesses_gpu = None
    composite_image_gpu = None

    PY = False
    GPU = True
    DEBUG_ALL = False
    EVAL_UNTIL = 1
    RESIZE = 100
    MAX_THREADS = 1024

    radius = 1


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

    files = [f for f in os.listdir(str(path)) if any(f.lower().endswith(ending) for ending in FILE_EXTENTIONS["RAW"] + FILE_EXTENTIONS["CV2"])]
    print(files)

    images_lst = []
    for i, f in enumerate(files):
        print("Loading", f, i)
        if i == EVAL_UNTIL:
            break

        if any(f.lower().endswith(ending) for ending in FILE_EXTENTIONS["CV2"]):
            bgr = cv2.imread(str(path.joinpath(f)))

        elif any(f.lower().endswith(ending) for ending in FILE_EXTENTIONS["RAW"]):
            with rawpy.imread(str(path.joinpath(f))) as raw:
                rgb = raw.postprocess(use_camera_wb=True)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                print(bgr.flatten(order="K")[:5], bgr.flatten(order="K")[-5:])

        if RESIZE != 100:
            scale_percent = RESIZE  # percent of original size
            width = int(bgr.shape[1] * scale_percent / 100)
            height = int(bgr.shape[0] * scale_percent / 100)
            dim = (width, height)
            bgr = cv2.resize(bgr, dim, interpolation=cv2.INTER_AREA)
        else:
            width = int(bgr.shape[1])
            height = int(bgr.shape[0])

        images_lst.append(bgr)
        print("Loaded:", bgr.shape, bgr.dtype)
    
    for i,bgr in enumerate(images_lst):
        print(f"Evaluating {i}")
        start = time.time()

        if PY:
            ## algo
            if i == 0:
                composite_image = copy.deepcopy(bgr)
                sharpnesses = np.zeros(shape=(width, height), dtype=float)

                for x in range(width):
                    for y in range(height):
                        sharpnesses[x][y] = get_sharpness(bgr, x, y, radius)

            else:
                for x in range(width):
                    for y in range(height):
                        sh = get_sharpness(bgr, x, y, radius)

                        if sh > sharpnesses[x][y]:
                            sharpnesses[x][y] = sh
                            composite_image[y][x] = bgr[y][x]

        if GPU:
            if i == 0:
                composite_image_gpu = np.zeros((width * height * 3), dtype=np.uint8)
                sharpnesses_gpu = np.zeros((width * height), dtype=float)

            total_pixels = width*height

            grid_width = find_nearest_pow_2(total_pixels/MAX_THREADS)
            
            compareAndPushSharpnesses(
                drv.InOut(composite_image_gpu), drv.InOut(sharpnesses_gpu), drv.In(bgr.flatten(order="K")),
                drv.In(np.array([width])), drv.In(np.array([height])), drv.In(np.array([radius])),
                block=(MAX_THREADS, 1, 1), grid=(grid_width, 1))
            
            print(composite_image_gpu.shape)
            scale_percent_prog = 30  # percent of original size
            width_prog = int(bgr.shape[1] * scale_percent_prog / 100)
            height_prog = int(bgr.shape[0] * scale_percent_prog / 100)
            cv2.imshow("image gpu", cv2.resize(cv2.rotate(composite_image_gpu.reshape(height, width, 3), cv2.ROTATE_90_CLOCKWISE), (height_prog, width_prog)))
            cv2.moveWindow("image gpu", 100, 100)
            if i == 0:
                cv2.waitKey(0)
            else:
                cv2.waitKey(200)
            
        print("Time to eval:", time.time() - start)



    ## only when flattened
    if GPU:
        composite_image_gpu = composite_image_gpu.reshape(height, width, 3)
        # sharpnesses_gpu = sharpnesses_gpu.reshape(width, height)
        # sharpnesses_gpu = cv2.rotate(cv2.flip(sharpnesses_gpu, 0), cv2.ROTATE_90_CLOCKWISE)


    # if PY:
    #     sharpnesses = cv2.rotate(cv2.flip(sharpnesses, 0), cv2.ROTATE_90_CLOCKWISE)


    cv2.imwrite("out.png", cv2.blur(composite_image_gpu, (5,5)))


    if DEBUG_ALL:
        til_idx = 30
        if PY:
            print("\n".join([str(x) for x in nparr_to_arr(composite_image[:til_idx])]))
            print("\n")
        if GPU:
            print("\n".join([str(x) for x in nparr_to_arr(composite_image_gpu[:til_idx])]))
            print("\n")
        if PY:
            print("\n".join([str(x) for x in nparr_to_arr(sharpnesses[:til_idx], 2)]))
            print("\n")
        if GPU:
            print("\n".join([str(x) for x in nparr_to_arr(sharpnesses_gpu[:til_idx], 2)]))
            print("\n")

        print([f"{x}\n" for x in nparr_to_arr(composite_image[:til_idx])])
        print([f"{x}\n" for x in nparr_to_arr(composite_image_gpu[:til_idx])])
        print([f"{x}\n" for x in nparr_to_arr(sharpnesses[:til_idx], 2)])
        print([f"{x}\n" for x in nparr_to_arr(sharpnesses_gpu[:til_idx], 2)])

        if GPU:
            with open("sharpnesses_gpu.txt", "w") as wf:
                s = ""
                for y in sharpnesses_gpu:
                    for x in y:
                        s += f"{x:.7f} "

                    s += "\n"
                wf.write(s)

        if PY:
            with open("sharpnesses_py.txt", "w") as wf:
                s = ""
                for y in sharpnesses:
                    for x in y:
                        s += f"{x:.7f} "

                    s += "\n"
                wf.write(s)

        if PY:
            with open("image_data_py.txt", "w") as wf:
                s = ""
                for y in composite_image:
                    for x in y:
                        s += f"{hex(x[0])[-2:]}{hex(x[1])[-2:]}{hex(x[2])[-2:]} "

                    s += "\n"
                wf.write(s)

        if GPU:
            with open("image_data_gpu.txt", "w") as wf:
                s = ""
                for y in composite_image_gpu:
                    for x in y:
                        s += f"{hex(x[0])[-2:]}{hex(x[1])[-2:]}{hex(x[2])[-2:]} "

                    s += "\n"
                wf.write(s)

    if PY:
        cv2.imshow('image py', cv2.rotate(composite_image, cv2.ROTATE_90_CLOCKWISE))
        # cv2.imshow('sharpness', cv2.rotate(sharpnesses, cv2.ROTATE_90_CLOCKWISE))
    if GPU:
        cv2.imshow('image gpu', cv2.rotate(composite_image_gpu, cv2.ROTATE_90_CLOCKWISE))
        # cv2.imshow('sharpness gpu', cv2.rotate(sharpnesses_gpu, cv2.ROTATE_90_CLOCKWISE))

    # if GPU and PY:
    #     cv2.imshow('sharpness diff', sharpnesses_gpu - sharpnesses)

    cv2.waitKey(0)

if __name__ == '__main__':
    main()
