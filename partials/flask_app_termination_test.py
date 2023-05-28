import time
from flask import Flask, request
from threading import Thread
import werkzeug.serving as serving
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


def worker():
    while True:
        print("working")
        time.sleep(1)


global server


org_mk_server = serving.make_server
print(org_mk_server)

def wrapper(*args, **kwargs):
    global server
    server = org_mk_server(*args, **kwargs)
    print(server)
    return server

serving.make_server = wrapper

app = Flask(__name__)


global kill
kill = False

def checkkill():
    global kill
    global server
    while not kill:
        time.sleep(1)
    print(server)
    server.shutdown()


worker_thread = Thread(target=worker)


Thread(target=checkkill).start()


@app.route("/")
def index():
    worker_thread.start()
    return "<a href='/shutdown'>Terminate the app</a>"



@app.route('/shutdown', methods=['GET'])
def shutdown():
    global kill
    kill = True

    return 'Server shutting down...'

app.run()
