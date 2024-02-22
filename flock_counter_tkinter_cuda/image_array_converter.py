import cv2

def convert_gray_arr_to_image(arr, width, height):
    return cv2.cvtColor(cv2.flip(cv2.rotate(arr.reshape(width, height), cv2.ROTATE_90_CLOCKWISE), 1), cv2.COLOR_GRAY2RGB)


def convert_color_arr_to_image(arr, width, height):
    return cv2.cvtColor(arr.reshape(height, width, 3), cv2.COLOR_BGR2RGB)