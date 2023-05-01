from flask import Flask, Response
import time

app = Flask(__name__)

global stream
global bytes_
stream = open('Movie.mjpeg', 'rb')
bytes_ = b""
def get_frame():
    global stream
    global bytes_

    while True:
        bytes_ += stream.read(1024)
        a = bytes_.find(b'\xff\xd8')
        b = bytes_.find(b'\xff\xd9')
        time.sleep(.001)
        if a != -1 and b != -1:
            jpg = bytes_[a:b+2]
            bytes_ = bytes_[b+2:]
            yield (b'--frame\r\n'
                b'Content-Type: text/plain\r\n\r\n'+jpg+b'\r\n')
            
        elif a == b == -1:
            stream = open('Movie.mjpeg', 'rb')



@app.route('/vid')
def vid():
     return Response(get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='localhost',port=3000, debug=True, threaded=True)
    