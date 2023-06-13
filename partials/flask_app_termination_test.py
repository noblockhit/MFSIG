import time
from flask import Flask
import werkzeug.serving as serving

global server

org_mk_server = serving.make_server

def wrapper(*args, **kwargs):
    global server
    server = org_mk_server(*args, **kwargs)
    return server

serving.make_server = wrapper

app = Flask(__name__)


@app.route("/")
def index():
    return "<a href='/shutdown'>Terminate the app</a>"



@app.route('/shutdown', methods=['GET'])
def shutdown():
    global server
    server.shutdown()

    return 'Server shutting down...'

app.run()
