import rawpy
import cv2


def load_raw_image(path, brightness=1):
    with rawpy.imread(path) as raw:
        # Access the raw image data
        raw_data = raw.raw_image
        height, width = raw_data.shape
        raw_array = raw_data.reshape((height, width, 1))

        # Demosaic the raw array
        raw_image = cv2.convertScaleAbs(cv2.cvtColor(raw_array, 46), alpha=(255.0/65535.0) * 16 * brightness)
        return raw_image
