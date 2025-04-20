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

# min/max move threshold (cm)
min_move_cm = 20
max_move_cm = 80

objects = []

# Load COCO class names
with open('COCO/object_detection_classes_coco.txt', 'r') as f:
    class_names = f.read().split('\n')

COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))

model = cv2.dnn.readNet(model='COCO/frozen_inference_graph.pb',
                        config='COCO/ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt',
                        framework='TensorFlow')

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

def detection_center(det):
    """Center x, y normalized to [-0.5, 0.5]"""
    cx = (det[3] + det[5]) / 2.0 - 0.5
    cy = (det[4] + det[6]) / 2.0 - 0.5
    return (cx, cy)

def norm(vec):
    return np.sqrt(vec[0]**2 + vec[1]**2)

def closest_detection(detections, old_det):
    best_det = None
    min_dist = float('inf')
    old_center = detection_center(old_det)
    for det in detections[0, 0]:
        if det[2] > 0.4 and int(det[1]) == 1:  # 'person'
            center = detection_center(det)
            distance = norm((center[0] - old_center[0], center[1] - old_center[1]))
            if distance < min_dist:
                min_dist = distance
                best_det = det
    return best_det


def clip(val):
    if abs(val) < min_move_cm:
        return 0
    return max(-max_move_cm, min(max_move_cm, val))

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


# Large updates: 
def generate_frames():
    first_frame = True
    while True:
        with thread_lock:
            try:
                global model 
                global objects
                tello = get_tello()
                if tello is not None:
                    frame_read = tello.get_frame_read()
                    if frame_read is not None and frame_read.frame is not None:
                        frame = frame_read.frame
                        # Convert BGR to RGB
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        # Resize frame for better performance
                        frame = cv2.resize(frame, (640, 480))
                        h, w, _ = frame.shape
                        # Object detection
                        blob = cv2.dnn.blobFromImage(frame, size=(300, 300), mean=(104, 117, 123), swapRB=True)
                        model.setInput(blob)
                        detections = model.forward()
                        #print(first_frame)
                        if first_frame: 
                            print("enterd")
                            id_num = 1
                            # All detections need to be above threshold
                            #print(detections[0,0])
                            probs = [det[2] for det in detections[0,0]]
                            print(probs)
                            if np.all(np.array(probs) <= 0.4):
                                print('none')
                                pass
                            else:
                                for det in detections[0,0]:
                                    if det is not None:
                                        if det[2] > 0.4 and int(det[1]) == 1:
                                            class_id = int(det[1])
                                            class_name = class_names[class_id - 1]
                                            color = COLORS[class_id]

                                            x1 = int(det[3] * w)
                                            y1 = int(det[4] * h)
                                            x2 = int(det[5] * w)
                                            y2 = int(det[6] * h)
                                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                                            label = f"{class_name} {det[2]:.2f}"

                                            data = {
                                                "id": id_num,
                                                "det": det,
                                                "bbox": {
                                                    "x1": x1,
                                                    "y1": y1,
                                                    "x2": x2,
                                                    "y2": y2
                                                }
                                            }
                                            objects.append(data)   
                                            id_num += 1                             
                                print('reached initial')
                                print(len(objects))
                                first_frame = False
                        else:
                            print('thru objects')
                            for obj in objects:
                                id_num = obj["id"]
                                old_det = obj["det"]
                                old_bbox = obj["bbox"]
                                new_det = closest_detection(detections, old_det)
                                x1 = int(new_det[3] * w)
                                y1 = int(new_det[4] * h)
                                x2 = int(new_det[5] * w)
                                y2 = int(new_det[6] * h)
                                new_det_bbox = {
                                    "x1": x1,
                                    "y1": y1,
                                    "x2": x2,
                                    "y2": y2
                                }
                                if new_det is not None:
                                    obj["det"] = new_det
                                    obj["bbox"] = new_det_bbox
                    for obj in objects:
                        print(objects)
                        det = obj["det"]
                        id_num = obj["id"]
                        bbox = obj["bbox"]
                        class_id = int(det[1])
                        class_name = class_names[class_id - 1]
                        color = COLORS[class_id]

                        x1 = bbox["x1"]
                        y1 = bbox["y1"]
                        x2 = bbox["x2"]
                        y2 = bbox["y2"]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        label = f"{id_num} {det[2]:.2f}"
                        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        print("Drawing object ", id_num)
                        #print('drawing frames')

                else:
                    frame = create_dummy_frame()
                    print("here2")
                
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