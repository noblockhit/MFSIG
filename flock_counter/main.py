import multiprocessing as mp
from tkinter import filedialog
import time
import cv2
from rawloader import load_raw_image
from display3d import show as show3d
import pickle


color_palette = [
    0x3366cc,
    0xdc3912,
    0xff9900,
    0x109618,
    0x990099,
    0x0099c6,
    0xdd4477,
    0x66aa00,
    0xb82e2e,
    0x316395,
    0x994499,
    0x22aa99,
    0xaaaa11,
    0x6633cc,
    0xe67300,
    0x8b0707,
    0x651067,
    0x329262,
    0x5574a6,
    0x3b3eac,
    0xb77322,
    0x16d620,
    0xb91383,
    0xf4359e,
    0x9c5935,
    0xa9c413,
    0x2a778d,
    0x668d1c,
    0xbea413,
    0x0c5922,
    0x743411
]

def hex_to_rgb(h):
    return tuple(int(str(h)[i:i+2], 16) for i in (0, 2, 4))

colors = [hex_to_rgb(c) for c in color_palette]


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
        rgb = load_raw_image(name, 35)

    if rgb.shape[0] > rgb.shape[1]:
        rgb = cv2.rotate(rgb, cv2.ROTATE_90_CLOCKWISE)

    return rgb

imgs = {}

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
    print(image_paths)
    rgb_values = mp.Pool(min(60, len(image_paths))).imap(load_image, image_paths)

    for idx, (name, rgb) in enumerate(zip(image_paths, rgb_values)):
        print("loaded", name)
        imgs[name] = rgb

    loading_time = (time.time_ns() - img_load_time_start) / (10 ** 9) 


def key_out_tips(img_in):
    blurred = cv2.medianBlur(img_in, 7)
    mask = cv2.cvtColor(cv2.threshold(blurred, 240, 1, cv2.THRESH_BINARY)[1], cv2.COLOR_RGB2GRAY)

    img = cv2.bitwise_and(img_in, img_in, mask=mask)
    edged = cv2.Canny(image=cv2.cvtColor(mask*255, cv2.COLOR_GRAY2BGR), threshold1=140, threshold2=255)
    return img, edged


def rect_is_same(rect1, rect2):
    _, x1, y1, w1, h1 = rect1
    _, x2, y2, w2, h2 = rect2

    # Check for overlap
    if (x1 < x2 + w2 and x1 + w1 > x2 and
        y1 < y2 + h2 and y1 + h1 > y2):
        return True
    else:
        return False


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
                (rect_x,rect_y),(w,h),angl = cv2.minAreaRect(cntr)
                # x = rect_x + w//2
                # y = rect_y + h//2
                # radius = (w+h)//4
                (x,y),radius = cv2.minEnclosingCircle(cntr)
                x, y, radius = int(x), int(y), int(radius)

                
                if w > 37 < h and w < 180 > h:
                    if 28 < radius < 90:
                        # cv2.drawContours(result, [cntr], 0, colors[temp_col_idx % len(colors)], 4)
                        # temp_col_idx += 1
                        # cv2.circle(result, (x, y), radius, (255,0,0), 2)
                        circles.append([image_name, x, y, radius, idx])
                                

        # qs.show(f"res {image_name}", result)
        # cv2.waitKey(0)
        return circles
    except TypeError:
        return [] 

def main():
    on_load_new_image()

    circles_list = mp.Pool(min(60, len(imgs))).imap(generate_circles, list(enumerate(imgs.items())))

    circles = []
    for cs in circles_list:
        circles += cs
        if len(cs) > 0:
            print(f"evaluated {cs[0][0]}")

    plot_data_x, plot_data_y, plot_data_z = zip(*[(center_x, center_y, image_idx) for image_name, center_x, center_y, radius, image_idx in circles])

    lines = []
    lines.append(list(range(0,len(circles))))


    pairs = [-1]*len(circles)
    for x, p in enumerate(pairs):
        dist = 10*10**9
        for y in lines[0]:
            new_dist = (plot_data_x[x]-plot_data_x[y])**2 + (plot_data_x[x]-plot_data_y[y])**2
            print(f"{x:02d}, {y:02d}, {new_dist:09d}")
            if new_dist < dist:
                pairs[x] = y
            dist = new_dist

    lines.append(pairs)
    print(lines)


    show3d((plot_data_x, plot_data_y, plot_data_z))
    with open(b"tmp.3dgraph", "wb") as wf:
        pickle.dump({"circles":(plot_data_x, plot_data_y, plot_data_z), "lines": lines}, wf)
    input()

    # for idx, c1 in enumerate(circles):
    #     if c1[4] != 0:
    #         continue
    #     print(f"filtering cricle {idx+1}/{len(circles)}")
    #     for c2 in circles[idx:]:
    #         if c2[4] != 0:
    #             continue
    #         if c1 == c2:
    #             continue

    #         if (c1[1] - c2[1])**2+(c1[2] - c2[2])**2 < (c1[3] + c2[3])**2:
    #             if c1[3] > c2[3]:### delete larger
    #                 c1[0] = None
    #             else:
    #                 c2[0] = None
    
    # circles = [c for c in circles if c[0] != None]


    # for name, img in imgs.items():
    #     for idx, (image_name, center_x, center_y, radius, image_idx) in enumerate(circles):
    #         # cv2.putText(img, f"{idx}", (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 3, 255, 4)
    #         # if image_name == name:
    #         #     cv2.circle(img, (center_x, center_y), radius, (0,0,255), 6)
    #         #     circle_count += 1
    #         # else:
    #         #     cv2.circle(img, (center_x, center_y), radius, (255,255,255), 4)


    #         #     cv2.rectangle(img, (x, y), (x + w, y + h), (0,0,255), 10)
    #         # else:
    #         #     cv2.rectangle(img, (x, y), (x + w, y + h), (255,255,255), 8)
        
                
    #     # qs.show("image", img)
    #     # cv2.waitKey(200)


if __name__ == "__main__":
    main()
