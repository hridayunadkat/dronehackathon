from flask import Flask, Response
from flask_socketio import SocketIO
import cv2
import numpy as np
import base64
import threading
import time
from djitellopy import Tello

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

tello = None
thread = None
thread_lock = threading.Lock()

def get_tello():
    global tello
    if tello is None:
        try:
            tello = Tello()
            tello.connect()
            tello.streamon()
            print("Tello connected successfully")
        except Exception as e:
            print(f"Error connecting to Tello: {e}")
            return None
    return tello

def create_dummy_frame():
    # Create a black frame with text
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    text = "No Tello Feed Available"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_x = (frame.shape[1] - text_size[0]) // 2
    text_y = (frame.shape[0] + text_size[1]) // 2
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)
    return frame

def generate_frames():
    while True:
        with thread_lock:
            try:
                tello = get_tello()
                if tello is not None:
                    frame_read = tello.get_frame_read()
                    if frame_read is not None and frame_read.frame is not None:
                        frame = frame_read.frame
                        # Convert BGR to RGB
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        # Resize frame for better performance
                        frame = cv2.resize(frame, (640, 480))
                    else:
                        frame = create_dummy_frame()
                else:
                    frame = create_dummy_frame()
                
                # Convert frame to JPEG
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                # Convert to base64 for WebSocket transmission
                frame_base64 = base64.b64encode(frame).decode('utf-8')
                socketio.emit('video_frame', {'frame': frame_base64})
            except Exception as e:
                print(f"Error in frame generation: {e}")
                frame = create_dummy_frame()
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                frame_base64 = base64.b64encode(frame).decode('utf-8')
                socketio.emit('video_frame', {'frame': frame_base64})
        time.sleep(0.033)  # ~30 FPS

@socketio.on('connect')
def handle_connect():
    global thread
    with thread_lock:
        if thread is None:
            thread = socketio.start_background_task(generate_frames)

@app.route('/')
def index():
    return "Tello Camera Stream Server"

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5001, debug=True) 