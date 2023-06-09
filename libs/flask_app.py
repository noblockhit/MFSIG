from flask import Flask, render_template, Response, request, send_from_directory
import werkzeug.serving as serving
from sbNative.runtimetools import get_path
import time
from PIL import Image
import io
from traceback import print_exc
from urllib.parse import unquote
from pathlib import Path
import os
import json
from libs import cameraParser
from libs.deps import bmscam
from .state import State

org_mk_server = serving.make_server


def wrapper(*args, **kwargs):
    State.server = org_mk_server(*args, **kwargs)
    return State.server


serving.make_server = wrapper

app = Flask(__name__,
            template_folder=str(get_path() / "deps" / "flask" / "templates"),
            static_folder=str(get_path() / "deps" / "flask" / "static"))


if str(get_path()) == ".":
    State.image_dir = Path(__file__).parent.parent / "images"
else:
    State.image_dir = get_path().parent / "images"

State.isGPIO = False

try:
    from . import gpio_handler
except ImportError:
    print_exc()
    print("An Import Error occured, this might be because of your device not having GPIO pins. In This case ignore this Error, otherwise inspect the traceback above.")
else:
    State.isGPIO = True
    State.motor = gpio_handler.Motor([21, 20, 19, 16])
    State.motor.calibrate()


def generate_live_image():
    while True:
        time.sleep(.1)

        try:
            pil_img = Image.frombytes(
                "RGB", (State.imgWidth, State.imgHeight), State.pData)
        except TypeError:
            yield (b'--frame\r\n'
               b'Content-Type: text\r\n\r\n' + b"It seems a temporary issue has occured..." + b'\r\n')
        img_byte_arr = io.BytesIO()
        pil_img.save(img_byte_arr, format="jpeg")

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + img_byte_arr.getvalue() + b'\r\n')


def reset_camera_properties():
    try:
        State.camera.Close()
    except AttributeError:
        pass

    State.imgWidth, State.imgHeight, State.pData = None, None, None

    State.curr_device = None
    State.resolution_idx = None
    State.camera = None
    State.microscope_start = 0
    State.microscope_end = 0
    State.microscope_position = 0
    State.real_motor_position = 0
    State.recording = False
    State.start_motor_and_prepare_recording_running = False
    State.image_count = 1
    State.recording_progress = None
    State.current_image_index = 0
    State.busy_capturing = False


reset_camera_properties()


@app.route("/")
def camera_select():
    return render_template("cameraselect.html")


@app.route("/liveview")
def liveview():
    if bool(request.args.get("with_bms_cam")) is False:
        State.with_bms_cam = True
    else:
        State.with_bms_cam = False
    ## set the camera
    
    if State.with_bms_cam:
        if (State.curr_device is None or 
            State.resolution_idx is None):
            return "One or more of the parameters necessary for the BMS camera have not been set, please return to the main page and choose to use another way of capturing an image or set all the necessary parameters!"
    
        if State.camera is not None:
            State.camera.Close()
        State.camera = cameraParser.bmscam.Bmscam.Open(State.curr_device.id)

        if State.camera:
            State.camera.put_eSize(State.resolution_idx)
            State.resolution_idx = State.camera.get_eSize()

            State.imgWidth = State.curr_device.model.res[State.resolution_idx].width
            State.imgHeight = State.curr_device.model.res[State.resolution_idx].height

            State.camera.put_Option(
                cameraParser.bmscam.BMSCAM_OPTION_BYTEORDER, 0)
            State.camera.put_AutoExpoEnable(1)

            State.pData = bytes(cameraParser.bmscam.TDIBWIDTHBYTES(
                State.imgWidth * 24) * State.imgHeight)

            try:
                State.camera.StartPullModeWithCallback(
                    cameraParser.event_callback, ())
            except cameraParser.bmscam.HRESULTException as e:
                print("Failed to start camera.", e)
                State.camera.Close()
    else:
        if State.isGPIO:
            State.camera = gpio_handler.Camera(26)

    return render_template("liveview.html")


@app.route("/stepsetter")
def stepsetter():
    return render_template("stepsetter.html")


@app.route('/favicon.svg')
def favicon():
    _path = get_path()
    if str(_path) == ".":
        _path = Path(__file__).parent
    return send_from_directory(str(_path / "deps" / "flask" / "static"),
                               'favicon.svg', mimetype='image/svg+xml')


@app.route("/image-count/<count>", methods=["POST"])
def set_image_count(count):
    State.image_count = int(float(count))
    return "", 200


@app.route("/cameras", methods=["GET"])
def get_cameras():
    ret = ""
    for idx, device in cameraParser.list_devices():
        ret += f'<option value="{idx}">{device.displayname}: {device.id}</option>\n'

    return ret.strip("\n")


@app.route("/camera/<camera_idx>", methods=["POST"])
def set_camera(camera_idx):
    if State.curr_device is not None:
        reset_camera_properties()

    State.curr_device = State.bms_enum[int(camera_idx)]

    ret = ""
    for idx, reso in cameraParser.get_current_devices_resolution_options(State.curr_device):
        ret += f'<option value="{idx}">{reso[0]} x {reso[1]}</option>\n'

    return ret.strip("\n")


@app.route("/resolution/<reso_idx>", methods=["POST"])
def set_resolution(reso_idx):
    State.resolution_idx = int(reso_idx)
    return "", 200


@app.route("/files/directory/list/<enc_directory>")
def list_directory(enc_directory):
    if enc_directory == "null":
        plib_dir = Path(State.image_dir)
    else:
        directory = unquote(unquote(enc_directory))
        plib_dir = Path(directory)

    ret = {}

    if plib_dir != plib_dir.parent:  # if plib_dir is not most parent folder
        ret[".."] = str(plib_dir.parent)

    for subfolder in os.listdir(plib_dir):
        if os.path.isdir(str(plib_dir / subfolder)):
            ret[subfolder] = str(plib_dir / subfolder)

    State.image_dir = plib_dir
    return json.dumps(ret)


@app.route("/files/directory/get")
def get_current_images_directory():
    return str(State.image_dir)


@app.route("/record-images")
def start_recording():
    if State.recording:
        return "Already started recording", 400

    # if State.camera is None:
    #     return "You have not selected a camera yet!", 400

    if State.image_count > State.microscope_end - State.microscope_start:
        return "You may not take more images than Steps taken by the motor, this is redundant due to having multiple images in the same position.", 400
    
    State.recording = True
    return "", 200


@app.route("/microscope/current")
def current_pos():
    return str(State.microscope_position)


@app.route("/microscope/move/<amount>")
def move_down(amount):
    State.microscope_position += int(amount)
    return str(State.microscope_position)


@app.route("/microscope/move/start", methods=["GET"])
def move_start():
    State.microscope_position = State.microscope_start
    return str(State.microscope_position)


@app.route("/microscope/move/end", methods=["GET"])
def move_end():
    State.microscope_position = State.microscope_end
    return str(State.microscope_position)


@app.route("/microscope/start", methods=["GET", "POST"])
def set_start():
    if request.method == "POST":
        State.microscope_start = State.microscope_position

    elif request.method == "CONNECT":
        State.microscope_position = State.microscope_start

    return str(State.microscope_start)


@app.route("/microscope/end",  methods=["GET", "POST"])
def set_end():
    if request.method == "POST":
        State.microscope_end = State.microscope_position

    return str(State.microscope_end)


@app.route("/live-stream")
def live_stream():
    if not State.imgWidth or not State.imgHeight or not State.pData:
        return Response("The camera has seemingly not been started yet", status=400)

    return Response(generate_live_image(), mimetype='multipart/x-mixed-replace; boundary=frame')
