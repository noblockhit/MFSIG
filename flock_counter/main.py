import rawpy
import multiprocessing as mp
from tkinter import filedialog
import time
import cv2


def show(name, data, x, y):
    ar = data.shape[1] / data.shape[0]
    height = 400
    width = int(ar*height)
    print(width, height)
    cv2.imshow(name, cv2.resize(data, (width, height)))
    cv2.moveWindow(name, x, y)
    
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
        with rawpy.imread(name) as raw:
            rgb = raw.postprocess(use_camera_wb=False)

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
    
    rgb_values = mp.Pool(len(image_paths)).imap(load_image, image_paths)

    for idx, (name, rgb) in enumerate(zip(image_paths, rgb_values)):
        imgs[name] = rgb

    loading_time = (time.time_ns() - img_load_time_start) / (10 ** 9) 


def key_out_tips(img_in):
    blurred = cv2.medianBlur(img_in, 7)
    mask = cv2.cvtColor(cv2.threshold(blurred, 155, 1, cv2.THRESH_BINARY)[1], cv2.COLOR_RGB2GRAY)

    img = cv2.bitwise_and(img_in, img_in, mask=mask)
    edged = cv2.Canny(image=img, threshold1=120, threshold2=200)
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


def main():
    on_load_new_image()
    
    print("done")

    rects = []
    for image_name, input_img in imgs.items():
        keyed, img = key_out_tips(input_img)
        # threshold
        thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)[1]

        # get contours
        contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        hierarchy = contours[1] if len(contours) == 2 else contours[2]
        contours = contours[0] if len(contours) == 2 else contours[1]

        # get the actual inner list of hierarchy descriptions
        hierarchy = hierarchy[0]

        # count inner contours

        result = img.copy()
        result = cv2.merge([result,result,result])
        for cntr, hier in zip(contours, hierarchy):
            if hier[3] == -1:
                try:
                    M = cv2.moments(cntr)
                    center_x = int(M["m10"] / M["m00"])
                    center_y = int(M["m01"] / M["m00"])
                    area = cv2.contourArea(cntr)
                    x,y,w,h = cv2.boundingRect(cntr)
                except ZeroDivisionError:
                    continue
                else:
                    if w > 75 < h and w < 220 > h:
                        print(w, h, area)
                        cv2.drawContours(result, [cntr], 0, (0,0,255), 2)
                        cv2.rectangle(result, (x, y), (x + w, y + h), (0,255,255), 1)
                        cv2.circle(result, (center_x, center_y), 2, (255,0,255), 2)
                        print(image_name, x, y, w, h)
                        rects.append([image_name, x, y, w, h])

        
        # show result
        cv2.destroyAllWindows()
        show("input img", input_img, 0, 30)
        show("keyed img", keyed, 400, 30)
        show("result", result, 800, 30)
        cv2.waitKey(2000)
        print(f"finished calc for {image_name}")
    cv2.destroyAllWindows()
    
    for idx, r1 in enumerate(rects):
        for r2 in rects:
            if r1 == r2:
                continue

            if rect_is_same(r1, r2):
                if r1[3] * r1[4] < r2[3] * r2[4]:### delete larger rectangle
                    r1[0] = None
                else:
                    r2[0] = None
    
    rects = [r for r in rects if r[0] != None]


    while True:
        for name, img in imgs.items():
            for image_name, x, y, w, h in rects:
                if image_name == name:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0,0,255), 6)
                else:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255,255,255), 4)
            
            
            show("image", img, 50, 50)
            cv2.waitKey(200)

            


if __name__ == "__main__":
    main()
