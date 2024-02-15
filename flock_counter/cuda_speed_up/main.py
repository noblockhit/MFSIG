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
from sbNative.runtimetools import get_path
from slider import Slider
import plotly.graph_objects as go


global KEYING_BLUR_RADIUS
global KEYING_MASK_THRESHHOLD
global CANNY_THRESHHOLD_1
global MIN_RADIUS
global MAX_RADIUS
global IMAGE_DISTANCE_TO_PIXEL_FACTOR
global MAXIMUM_DISTANCE_TO_SAME_TIP
global MAX_CORES_FOR_MP
global PREVIEW_IMAGE_HEIGHT


KEYING_BLUR_RADIUS = 1
KEYING_MASK_THRESHHOLD = 180
CANNY_THRESHHOLD_1 = 140
MIN_RADIUS = 25
MAX_RADIUS = 90
IMAGE_DISTANCE_TO_PIXEL_FACTOR = 20
MAXIMUM_DISTANCE_TO_SAME_TIP = 140
MAX_CORES_FOR_MP = mp.cpu_count()-1
PREVIEW_IMAGE_HEIGHT = 1000


class Checkpoint:
    time = time.perf_counter_ns()
    def __init__(self, *args, **kwds):
        delta = time.perf_counter_ns() - Checkpoint.time
        Checkpoint.time = time.perf_counter_ns()
        print(f"{delta*10**-9:.5f} seconds for {' '.join(args)} {str(kwds)[1:-1]}")


colors = [
    (255, 0  , 0  ),
    (0  , 255, 0  ),
    (0  , 0  , 255),
    
    (255, 255, 0  ),
    (0  , 255, 255),
    
    (255  , 255, 0  ),
    (255  , 0  , 255),
    
    (255, 0  , 255  ),
    (0  , 255, 255  ),
]


def find_nearest_pow_2(val):
    for i in range(32):
        if val < 2**i:
            return 2**i


class TipFinderCuda:
    def __init__(self):
        drv.init()
        dev = drv.Device(0)
        ctx = dev.make_context()

        with open(str(get_path() / "outlineTips.cu")) as f:
            self.mod = SourceModule(f.read())

        self.available_methods = ["outline_tips_method_1", "outline_tips_method_3"]
        self._current_method = 0
        self.current_method = 0
        
    @property
    def current_method_name(self):
        return self.available_methods[self.current_method]
    
    @property
    def current_method(self):
        return self._current_method
    
    
    @current_method.setter
    def current_method(self, value):
        self._current_method = value
        self.outline_tips = self.mod.get_function(self.available_methods[self._current_method])
    
        
    def find(self, input_img, idx, name):
        img = input_img.copy()
        height, width = img.shape[:2]
        
        
        out = np.zeros(shape=img.shape[:2], dtype=img.dtype)

        processed_input = cv2.blur(img, (3, 3))
        
        self.outline_tips(drv.In(processed_input), drv.InOut(out), drv.In(np.array(img.shape)),
                        block=(32, 32, 1), grid=(width//32+1, height//32+1))
        contours, hierarchy = cv2.findContours(out, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        # for i, contour in enumerate(contours):
        #     color = colors[i % len(colors)]
        #     cv2.drawContours(img, [contour], -1, color, 2)
        
        color_idx = 0
        tips = []
        print(len(contours))
        for cntr in contours:
            contour_points = cntr.squeeze()
            
            if 50 < len(contour_points) < 400:
                M = cv2.moments(cntr)
                if M["m00"] == 0:
                    continue
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                ## image_name, center_x, center_y, radius, image_idx
                tips.append((name, cX, cY, 69, idx))
                cv2.circle(img, (cX, cY), 9, (0, 0, 0), -1)
                cv2.circle(img, (cX, cY), 5, (255, 255, 255), -1)
                for p in contour_points:
                    img[p[1]][p[0]] = colors[color_idx]
                    # img[p[1]][p[0]+1] = [0, 255, 255]
                    # img[p[1]+1][p[0]] = [0, 255, 255]
                    # img[p[1]+1][p[0]+1] = [0, 255, 255]

                color_idx += 1
                color_idx = color_idx % len(colors)
        return img, out, tips


class ImageManager:
    def __init__(self):
        self.RAW_EXTENSIONS = [
                ".nef",
                ".arw",
                ".raw",
            ]
        
        self.CV2_EXTENSIONS = [
                ".jpeg",
                ".jpg",
                ".png",
                ".tiff",
                ".tif"
            ]
        
        self.imgs = collections.OrderedDict({})
        self.finder = TipFinderCuda()
        
    @property
    def supported_extensions(self):
        return self.RAW_EXTENSIONS + self.CV2_EXTENSIONS
    
    def _load(self, filepath):
        extension = f".{filepath.split('.')[-1].lower()}"
        if extension not in self.supported_extensions:
            raise ValueError(f"Invalid extension! Expected {self.supported_extensions} but found {extension}")
        
        if extension in self.RAW_EXTENSIONS:
            image = load_raw_image(filepath, 50)
        
        if extension in self.CV2_EXTENSIONS:
            image = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
            
        if image.shape[0] > image.shape[1]:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        return image
    
    
    def ask_and_load(self):
        selected_img_files = filedialog.askopenfiles(title="Open Images for the render queue",
                                                     filetypes=[("Image-files", " ".join(self.supported_extensions))])
        if not selected_img_files:
            return
        
        for filepath in selected_img_files:
            self.imgs[filepath.name] = self._load(filepath.name)
    
    
    def compute(self, name_or_image):
        if name_or_image in self.imgs.keys():
            image = self.imgs[name_or_image]
            name = name_or_image
            
        elif name_or_image in self.imgs.values():
            image = name_or_image
            for k, v in self.imgs.items():
                if v is image:
                    name = k
        else:
            raise ValueError("Unknown image")
        idx = list(self.imgs.keys()).index(name)
        return self.finder.find(image, idx, name)
    

    def draw_clusters(self, name_or_image, clusters):
        if name_or_image in self.imgs.keys():
            image = self.imgs[name_or_image]
            name = name_or_image
            
        elif name_or_image in self.imgs.values():
            image = name_or_image
            for k, v in self.imgs.items():
                if v is image:
                    name = k
        else:
            raise ValueError("Unknown image")
        img = image.copy()
        for cluster in clusters:
            cv2.drawContours(img, [np.array(cluster.points)], 0, (0, 0, 255), 2)

        return img

def make_circles(img_manager):
    satisfied = False
    shown_index = 0
    
    change = True
    while not satisfied:
        shown_index = (shown_index + len(img_manager.imgs)) % len(img_manager.imgs)
        shown_image_name = list(img_manager.imgs.keys())[shown_index]
        
        if change:
            preview_image, keyed_image, tips = img_manager.compute(shown_image_name)
        
            cv2.imshow("KI", keyed_image)
            cv2.imshow("IP", preview_image)
            cv2.moveWindow("KI", 50, 50)
            cv2.moveWindow("IP", 50, 50)
            info = f"{img_manager.finder.current_method} {shown_index}"
            cv2.setWindowTitle("KI", f"Keyed Image {info}")
            cv2.setWindowTitle("IP", f"Image Preview {info}")
            
            
        
        
        change = True
        key_ord = cv2.waitKey(1)
        

        if key_ord == ord("q"):
            shown_index -= 1
        elif key_ord == ord("e"):
            shown_index += 1
        elif key_ord == ord("1"):
            img_manager.finder.current_method = 0
        elif key_ord == ord("2"):
            img_manager.finder.current_method = 1
            
            
        elif key_ord == ord("s"):
            satisfied = True
            cv2.destroyAllWindows()
            break
        else:
            change = False
        
    all_tips = []
    for img in img_manager.imgs:
        _, _, tips = img_manager.compute(img)
        all_tips += tips
    
        
    return all_tips



def line_generator_gpu(circles_x_coords, circles_y_coords, circles_z_coords, radiuses, lines):
    global KEYING_BLUR_RADIUS
    global KEYING_MASK_THRESHHOLD
    global CANNY_THRESHHOLD_1
    global MIN_RADIUS
    global MAX_RADIUS
    global IMAGE_DISTANCE_TO_PIXEL_FACTOR
    global MAXIMUM_DISTANCE_TO_SAME_TIP
    global MAX_CORES_FOR_MP
    global PREVIEW_IMAGE_HEIGHT

    drv.init()
    dev = drv.Device(0)
    ctx = dev.make_context()
    Checkpoint("made context")

    with open(str(get_path() / "createLines.cu")) as f:
        mod = SourceModule(f.read())

    create_lines = mod.get_function("createLines")
    MAX_THREADS = 1024
    
    Checkpoint("Loading module")

    grid_width = find_nearest_pow_2(len(circles_x_coords)/MAX_THREADS)
    lines_x = np.empty(len(circles_x_coords)*3, dtype=np.int32)
    lines_y = np.empty(len(circles_y_coords)*3, dtype=np.int32)
    lines_z = np.empty(len(circles_z_coords)*3, dtype=np.int32)

    Checkpoint("allocating memory")
    
    lines_x[:] = -1
    lines_y[:] = -1
    lines_z[:] = -1
    
    Checkpoint("assigning init data")
    
    try:
        create_lines(
            drv.In(np.array(circles_x_coords, dtype=np.int32)), drv.In(np.array(circles_y_coords, dtype=np.int32)), drv.In(np.array(circles_z_coords, dtype=np.int32)),
            drv.In(np.array([len(circles_x_coords)])), drv.In(np.array([MAXIMUM_DISTANCE_TO_SAME_TIP*IMAGE_DISTANCE_TO_PIXEL_FACTOR])), drv.In(np.array([IMAGE_DISTANCE_TO_PIXEL_FACTOR])),
            drv.InOut(lines_x), drv.InOut(lines_y), drv.InOut(lines_z),
            block=(MAX_THREADS, 1, 1), grid=(grid_width, 1))
    except:
        ctx.pop()
        raise
    else:
        ctx.pop()
    
    Checkpoint("creating lines")

    lines[0] = [x if x >= 0 else None for x in lines_x]
    lines[1] = [y if y >= 0 else None for y in lines_y]
    lines[2] = [z if z >= 0 else None for z in lines_z]
    
    Checkpoint("filtering lines")
    
class Cluster:
    point_cluster_dict = {}
    clusters = []
    name_counter = 0
    def __init__(self, a, b):
        self.points = [a, b]
        self.name = Cluster.name_counter
        Cluster.name_counter += 1
        Cluster.clusters.append(self)
        Cluster.point_cluster_dict[a] = self
        Cluster.point_cluster_dict[b] = self

    def add_point(self, p):
        self.points.append(p)
        Cluster.point_cluster_dict[p] = self

    def merge(self, other):
        for p in other.points:
            if p not in self.points:
                self.points.append(p)
            Cluster.point_cluster_dict[p] = self

        Cluster.clusters.remove(other)
    
    @classmethod
    def add_new_line(cls, l):
        a, b = l
        cluster_from_a = Cluster.point_cluster_dict.get(a)
        cluster_from_b = Cluster.point_cluster_dict.get(b)

        if cluster_from_a is None and cluster_from_b is None:
            Cluster(a, b)
        
        elif cluster_from_a is not None and cluster_from_b is None:
            cluster_from_a.add_point(b)

        elif cluster_from_b is not None and cluster_from_a is None:
            cluster_from_b.add_point(a)
        
        elif cluster_from_a is cluster_from_b:
            pass
        else:
            cluster_from_a.merge(cluster_from_b)
    
    
    # def __repr__(self) -> str:
    #     return f"Cluster<{self.points}>"
    
    def __repr__(self) -> str:
        return f"Cluster <{self.name}>"
        

def count(lines=None):    
    if lines is None:
        lines = [[], [], []]

    print(f"There are {len(lines[0])} lines")
    z_lines = [((lines[0][i], lines[1][i], lines[2][i]), (lines[0][i+1], lines[1][i+1], lines[2][i+1])) for i in range(0, len(lines[0]), 3)]
    for idx, line in enumerate(z_lines):
        if line == ((None, None, None), (None, None, None)):
            continue
        Cluster.add_new_line(line)


def confirm(img_manager: ImageManager):
    shown_index = 0
    change = True
    while True:
        shown_index = (shown_index + len(img_manager.imgs)) % len(img_manager.imgs)
        shown_image_name = list(img_manager.imgs.keys())[shown_index]
        
        if change:
            image_with_clusters = img_manager.draw_clusters(shown_image_name, Cluster.clusters)
        
            cv2.imshow("IWC", image_with_clusters)
            cv2.moveWindow("IWC", 50, 50)
            info = f"{shown_index}"
            cv2.setWindowTitle("IWC", f"Image with Clusters {info}")
        
        
        change = True
        key_ord = cv2.waitKey(1)
        

        if key_ord == ord("q"):
            shown_index -= 1
        elif key_ord == ord("e"):
            shown_index += 1
            
            
        elif key_ord == ord("s"):
            break
            cv2.destroyAllWindows()
            break
        else:
            change = False

def main():
    Checkpoint(meta = "start")
    img_manager = ImageManager()
    img_manager.ask_and_load()
    Checkpoint("loaded")
    circles = make_circles(img_manager)
    Checkpoint("made circles")
    circles_x_coords, circles_y_coords, circles_z_coords, radiuses = zip(*[(center_x, center_y, image_idx, radius) for image_name, center_x, center_y, radius, image_idx in circles])
    Checkpoint("unzipped data")

    lines = [
        [],
        [],
        []
    ]
    line_generator_gpu(circles_x_coords, circles_y_coords, circles_z_coords, radiuses, lines)
    Checkpoint("generated lines")
    
    count(lines)
    Checkpoint("grouped and counted lines")
    confirm(img_manager)
    markers = []
    num = 0
    for c in Cluster.clusters:
        num += 1
        unzipped_points = list(zip(*c.points))
        marker_data = go.Scatter3d(
            x=unzipped_points[0], 
            y=unzipped_points[1], 
            z=unzipped_points[2], 
            marker=go.scatter3d.Marker(size=3), 
            opacity=0.8, 
            mode='markers'
        )
        markers.append(marker_data)
    Checkpoint("made graph")
    fig=go.Figure(data=markers)
    fig.write_html('grouped_points.html', auto_open=True)

    print(num)
    


if __name__ == "__main__":
    main()
