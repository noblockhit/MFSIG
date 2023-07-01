import time
import datetime
import sys
import os
import threading
from libs import flask_app
from libs.state import State


def start_motor_and_prepare_recording():
    if State.start_motor_and_prepare_recording_running:
        return
    State.start_motor_and_prepare_recording_running = True

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

    if State.with_bms_cam:
        now = datetime.datetime.now()
        formated_datetime = now.strftime("%Y_%m_%d_at_%H_%M_%S")

        State.final_image_dir = State.image_dir / f"BMSCAM_Images_from_{formated_datetime}"
        os.mkdir(str(State.final_image_dir))

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

    State.camera.Snap(0)
    
    # start recording
    target_total_steps = State.microscope_end - State.microscope_start
    avg_steps_per_image = target_total_steps / (State.image_count - 1)

    image_index = 0

    for step in range(target_total_steps):
        State.motor.step_forward()
        State.real_motor_position += 1
        if image_index * avg_steps_per_image > step:
            continue

        time.sleep(2)
        image_index += 1
        State.camera.Snap(0)

    State.camera.Close()
    sys.exit()

if __name__ == "__main__":
    th = threading.Thread(target=start_motor_and_prepare_recording)
    th.start()
    flask_app.app.run(host="0.0.0.0", port=80, threaded=True)
