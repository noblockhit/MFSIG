from flask import Flask, render_template
import socket
from threading import Thread
import websockets
from websockets.server import serve
import asyncio
import time
import random

global progress
progress = 0
async def respond(websocket):
    try:
        async for msg in websocket:
            if msg != "null" and int(msg) == progress:
                while int(msg) == progress:
                    time.sleep(.1)
            await websocket.send(str(progress))
    except websockets.exceptions.ConnectionClosedError:
        pass

async def main():
    print("running")
    async with serve(respond, "localhost", 65432):
        await asyncio.Future()  # run forever


def start_ws():
    asyncio.run(main())


def worker():
    global progress
    for i in range(11):
        time.sleep(random.randint(2, 10) / 5)
        print(i)
        progress = i*10


Thread(target=start_ws).start()
Thread(target=worker).start()


app = Flask(__name__, template_folder="./website/templates/")

@app.route("/", methods = ["GET", "POST"])
def home():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port = 25565, threaded=True)
