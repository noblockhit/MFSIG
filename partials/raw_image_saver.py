import cv2
from PIL import Image
import imageio
from imageio.plugins.pillow import PillowPlugin
import numpy as np
import matplotlib.pyplot as plt



# a = np.array([
#     [[1,2,3], [4,5,6]],
#     [[7,8,9], [10,11,12]]
# ])

# print(np.concatenate(np.concatenate(a)))


cap = cv2.VideoCapture(0) # video capture source camera (Here webcam of laptop) 
ret, frame = cap.read()


print(frame.shape)
# with open("image.raw", "wb") as wf:
#     b = np.concatenate(np.concatenate(frame)).tobytes()
#     wf.write(b)
#     print(np.concatenate(np.concatenate(frame)))

w,h = 3648, 2736

frame = np.full((w, h, 3), 255, dtype=np.uint8)

print(frame.shape)

pil_img = Image.fromarray(frame)
pil_img.save("image.tiff")


print("\n\n\n------------------------------------------------------------------------------------------------\n\n\n")

# with open("image_1685532650830.raw", "rb") as rf:
# print(np.fromfile("image_1685532650830.raw"))
