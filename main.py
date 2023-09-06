import time
import datetime
import sys
import os
import threading
from libs import flask_app
from libs.state import State
from flask import Flask, render_template
import socket
from threading import Thread
import websockets
from websockets.server import serve
import asyncio
import time
import random


def start_motor_and_prepare_recording():
    while True:
        if State.start_motor_and_prepare_recording_running:
            return
        State.start_motor_and_prepare_recording_running = True

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
                    time.sleep(.001)

        else:
            while True:
                if State.recording:
                    print("There was no GPIO detected, exiting program.")
                    break
                time.sleep(.1)

            ## this is for testing purposes only
            print("ENTERING TESTING PURPOSE ONLY SIMULATION, THERE WAS NO GPIO DETECTED")
            if State.microscope_start > State.microscope_end:
                State.microscope_end, State.microscope_start = State.microscope_start, State.microscope_end

            State.microscope_position = State.real_motor_position = State.microscope_start
            State.current_image_index = 0
            State.progress()
            
            # start recording
            target_total_steps = State.microscope_end - State.microscope_start
            avg_steps_per_image = target_total_steps / (State.image_count - 1)


            for step in range(target_total_steps):
                State.real_motor_position += 1
                if State.current_image_index * avg_steps_per_image > step:
                    continue

                time.sleep(2)
                State.current_image_index += 1
                State.busy_capturing = True
                State.progress()
                while State.busy_capturing:
                    time.sleep(.1)

        if State.isGPIO:
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
            State.current_image_index = 0

            State.camera.Snap(0)
            
            # start recording
            target_total_steps = State.microscope_end - State.microscope_start
            avg_steps_per_image = target_total_steps / (State.image_count - 1)


            for step in range(target_total_steps):
                State.motor.step_forward()
                State.real_motor_position += 1
                if State.current_image_index * avg_steps_per_image > step:
                    continue

                time.sleep(2)
                State.current_image_index += 1
                State.busy_capturing = True
                State.camera.Snap(0)
                while State.busy_capturing:
                    time.sleep(.1)

        State.start_motor_and_prepare_recording_running = False
        State.recording = False
        try:
            State.camera.Close()
        except AttributeError:
            pass


async def respond(websocket):
    try:
        async for msg in websocket:
            if not str(msg).isdigit():
                response_num = -1
            else:
                response_num = int(msg)

            while State.recording_progress is None or response_num == State.recording_progress:
                time.sleep(.1)
            await websocket.send(str(State.recording_progress))
    except websockets.exceptions.ConnectionClosedError:
        pass

async def websocket_coro():
    async with serve(respond, "0.0.0.0", 65432):
        await asyncio.Future()


def start_ws():
    asyncio.run(websocket_coro())

if __name__ == "__main__":
    th = threading.Thread(target=start_motor_and_prepare_recording)
    th.start()
    threading.Thread(target=start_ws).start()
    flask_app.app.run(host="0.0.0.0", port=80, threaded=True)
