from libs import cameraParser
from libs.cameraParser import bmscam
from libs import flask_app
import time
from PIL import Image
import io


def generate_live_image(imgWidth, imgHeight, pData):
    while True:
        time.sleep(.1)

        pil_img = Image.frombytes("RGB", (imgWidth, imgHeight), pData)

        img_byte_arr = io.BytesIO()
        pil_img.save(img_byte_arr, format="jpeg")


        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + img_byte_arr.getvalue() + b'\r\n')


if __name__ == "__main__":
    
    flask_app.app.get_live_image = generate_live_image
    flask_app.app.camparser = cameraParser
    flask_app.app.run(host="192.168.2.113", port=5000)
