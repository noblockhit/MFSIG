from flask import Flask, render_template, Response, request
from sbNative.runtimetools import get_path
import time
import numpy as np
from PIL import Image
import io

app = Flask(__name__,
            template_folder=str(get_path() / "deps" / "flask" / "templates"),
            static_folder=str(get_path() / "deps" / "flask" / "static"))


def complete_config():
    pass


@app.route("/")
def camera_select():
    return render_template("cameraselect.html")


@app.route("/cameras", methods=["GET"])
def get_cameras():
    ret = ""
    for idx, device in app.camparser.list_devices():
        ret += f'<option value="{idx}">{device.displayname}: {device.id}</option>\n'
        
    return ret.strip("\n")

@app.route("/camera/<camera_idx>", methods=["POST"])
def set_camera(camera_idx):
    curr_device = app.camparser.bms_enum[int(camera_idx)]
    
    ret = ""
    for idx, reso in app.camparser.get_current_devices_resolution_options(curr_device):
        ret += f'<option value="{idx}">{reso[0]} x {reso[1]}</option>\n'
        
    return ret.strip("\n")

@app.route("/liveview")
def liveview():
    return render_template("liveview.html")


@app.route("/live-stream")
def live_stream():
    return Response(app.get_live_image(),
        mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    def generate_example_live_image():
        image = np.zeros([680, 896, 3], dtype=np.uint8)
        image.fill(0)


        for tr in [[0,0], [0, 890], [674, 0], [674, 890]]:
            for x in range(5):
                for y in range(5):
                    image[tr[0]+x, tr[1]+y][0] = 255


        while True:
            time.sleep(.1)
            output = np.copy(np.array(image))

            pil_img = Image.fromarray(output)

            img_byte_arr = io.BytesIO()
            pil_img.save(img_byte_arr, format="jpeg")


            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + img_byte_arr.getvalue() + b'\r\n')
    
    app.get_live_image = generate_example_live_image
    app.run(host="192.168.2.113", port=5000)

