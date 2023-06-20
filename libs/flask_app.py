from flask import Flask, render_template, Response, request, send_from_directory
import werkzeug.serving as serving
from sbNative.runtimetools import get_path
import time
from PIL import Image
import io
import threading
from traceback import print_exc
from urllib.parse import unquote
from pathlib import Path
import os
import json
import datetime
import sys
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

        pil_img = Image.frombytes(
            "RGB", (State.imgWidth, State.imgHeight), State.pData)

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
    State.start_camera_and_motor_running = False
    State.image_count = 1


reset_camera_properties()


def start_camera_and_motor(*, with_bms_cam=True):
    if State.start_camera_and_motor_running:
        return
    State.start_camera_and_motor_running = True

    if with_bms_cam:
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

    print("IS GPIO:", State.isGPIO)
    if State.isGPIO:
        while True:
            if State.recording:
                break

            if State.real_motor_position < State.microscope_position:
                State.motor.step_forward()
                State.real_motor_position += 1

            elif State.real_motor_position > State.microscope_position:
                State.motor.step_backward()
                State.real_motor_position -= 1

            else:
                time.sleep(.5)

    else:
        while True:
            if State.recording:
                print("There was no GPIO detected, exiting program.")
                State.server.shutdown()
                exit()
            time.sleep(.5)

    if with_bms_cam:
        now = datetime.datetime.now()
        formated_datetime = now.strftime("%Y_%m_%d_at_%H_%M_%S")

<<<<<<< HEAD
        final_image_dir = State.image_dir / \
            f"BMSCAM_Images_from_{formated_datetime}"
        os.mkdir(str(final_image_dir))
=======
        State.final_image_dir = State.image_dir / f"BMSCAM_Images_from_{formated_datetime}"
        os.mkdir(str(State.final_image_dir))
>>>>>>> e2099f98ae5d58eb15acaf1a459e7ca146b1d4a8

    # making start smaller than end
    if State.microscope_start > State.microscope_end:
        State.microscope_end, State.microscope_start = State.microscope_start, State.microscope_end

    # moving to start position
    distance_to_start = State.microscope_start - State.real_motor_position
    if distance_to_start > 0:
        for _ in range(distance_to_start):
            State.motor.step_forward()
    elif distance_to_start < 0:
        for _ in range(-distance_to_start):
            State.motor.step_backward()

    State.microscope_position = State.real_motor_position = State.microscope_start

    time.sleep(3)

<<<<<<< HEAD
    State.camera.Snap()

=======
    State.camera.Snap(0)
    
>>>>>>> e2099f98ae5d58eb15acaf1a459e7ca146b1d4a8
    # start State.recording
    target_total_steps = State.microscope_end - State.microscope_start
    avg_steps_per_image = target_total_steps / (State.image_count - 1)

    image_index = 0

    for step in range(target_total_steps):
        State.motor.step_forward()
        State.real_motor_position += 1
<<<<<<< HEAD
        curr_amt_steps_taken += 1
        if (abs(image_index * avg_steps_per_image - curr_amt_steps_taken) >  # distance between the current image and the actual distance
                abs((image_index + 1) * avg_steps_per_image - curr_amt_steps_taken)):
=======
        print("stepped forward", image_index, avg_steps_per_image, step)
        if image_index * avg_steps_per_image > step:
>>>>>>> e2099f98ae5d58eb15acaf1a459e7ca146b1d4a8
            continue

        time.sleep(2)
        print("snapped")
        image_index += 1
        State.camera.Snap(0)

    sys.exit()


@app.route("/")
def camera_select():
    return render_template("cameraselect.html")


@app.route("/liveview")
def liveview():
    if bool(request.args.get("with_bms_cam")):
        th = threading.Thread(target=start_camera_and_motor,
                              kwargs={"with_bms_cam": 0})
        th.start()
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
    State.image_count = int(count)
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

    State.curr_device = cameraParser.bms_enum[int(camera_idx)]

    ret = ""
    for idx, reso in cameraParser.get_current_devices_resolution_options(State.curr_device):
        ret += f'<option value="{idx}">{reso[0]} x {reso[1]}</option>\n'

    return ret.strip("\n")


@app.route("/resolution/<reso_idx>", methods=["POST"])
def set_resolution(reso_idx):
    if State.resolution_idx is not None:
        temp_curr_device = State.curr_device
        reset_camera_properties()
        if temp_curr_device is not None:
            State.curr_device = temp_curr_device
            State.camera = cameraParser.bmscam.Bmscam.Open(
                State.curr_device.id)

    State.resolution_idx = int(reso_idx)
    th = threading.Thread(target=start_camera_and_motor)
    th.start()
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
    if not State.start_camera_and_motor_running:
        return "The motor (and optionally the camera) have not been started yet.", 400

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
        print("The camera has seemingly not been started yet")
        return Response("The camera has seemingly not been started yet", status=400)

    return Response(generate_live_image(), mimetype='multipart/x-mixed-replace; boundary=frame')
