from libs import flask_app


if __name__ == "__main__":
    flask_app.app.run(host="0.0.0.0", port=80, threaded=True)
