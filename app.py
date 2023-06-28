from heavynl.api import get_app, socketio

app = get_app()

if __name__ == "__main__":
    socketio.run(app)
