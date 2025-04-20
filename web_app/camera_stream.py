from flask import Flask, Response, request, jsonify
from flask_socketio import SocketIO
from flask_cors import CORS  # Import CORS
import cv2
import numpy as np
import base64
import threading
import time
import os
from djitellopy import Tello

app = Flask(__name__)
CORS(app)  # Enable CORS for the entire app
socketio = SocketIO(app, cors_allowed_origins="*")

tello = None
video_thread = None
video_thread_lock = threading.Lock()
controller_thread = None
controller_thread_lock = threading.Lock()


# min/max move threshold (cm)
min_move_cm = 20
max_move_cm = 80

print("Press 't' to takeoff, 'q' to land and quit.")
drone_in_air = False
kill_switch_engaged = False
autonomous_mode = True
circle_mode = False

area_target = 0.25
area_tolerance = 0.05
move_step_cm = 30
pixel_threshold = 40

last_move_time = time.time()
move_interval = 0.8

w, h = 320, 240

status = "Idle"

objects = []

frame = None  # Variable to store the current frame
frame_lock = threading.Lock()  # Lock for thread-safe access to frame
detected_box = None  # Variable to store the detected bounding box
detected_box_lock = threading.Lock()  # Lock for thread-safe access to detected_box

# Load COCO class names
with open('COCO/object_detection_classes_coco.txt', 'r') as f:
    class_names = f.read().split('\n')

COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))

model = cv2.dnn.readNet(model='COCO/frozen_inference_graph.pb',
                        config='COCO/ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt',
                        framework='TensorFlow')

clicked_coordinates = {"x": None, "y": None}  # Global variable to store coordinates

# Variable to store the last pressed key
last_pressed_key = None

def get_tello():
    global tello
    if tello is None:
        try:
            tello = Tello()
            tello.connect()
            tello.streamon()
            print("Tello connected successfully")
            print(f"Battery level: {tello.get_battery()}%")
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
            if distance < min_dist and det is not None:
                min_dist = distance
                best_det = det
    if best_det is None:
        best_det = old_det
    return best_det

def clip(val):
    if abs(val) < min_move_cm:
        return 0
    return max(-max_move_cm, min(max_move_cm, val))

def create_dummy_frame(text):
    # Create a black frame with text
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_x = (frame.shape[1] - text_size[0]) // 2
    text_y = (frame.shape[0] + text_size[1]) // 2
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)
    return frame

def go_in_circle(tello, angle, omega, radius):
    circle_time = angle/omega
    tello.send_rc_control(int(radius*np.deg2rad(angle)/(angle/omega)), 0, 0, int(-1*omega))
    #time.sleep(circle_time * 2)
    #tello.send_rc_control(0, 0, 0, 0)
    time.sleep(1)

def conv_ratio_to_radius(ratio):
    #y = -0.03x + 0.35
    # x = (1/-0.03)y + 0.35/0.03
    # alpha = -1/0.03
    # beta = 0.35/0.03
    # radius = alpha*ratio + beta
    beta = 5.76
    ft_to_cm = 30.48
    radius = ft_to_cm * np.sqrt(beta / ratio)
    return radius

def run_controller():
    global last_pressed_key
    global last_move_time
    global drone_in_air
    global tello
    global autonomous_mode
    global circle_mode
    global kill_switch_engaged
    global w
    global h
    global detected_box
    global detected_box_lock

    print(f"[{threading.current_thread().name}] Starting controller thread...")
    while True:
        try:
            # Get current time string
            current_time = time.strftime("%Y-%m-%d %H:%M:%S")
            try:
                if last_pressed_key == 'l' and drone_in_air:
                    last_pressed_key = None
                    tello.land()
                    status = "Drone has landed."
                    print(status)
                    break

                # elif last_pressed_key == '1' and not autonomous_mode:
                #     autonomous_mode = True
                #     mode = "AUTONOMOUS"
                #     status = f"Mode switched to {mode}"
                #     last_pressed_key = None
                #     print(status)
                #     continue

                # elif last_pressed_key == '2' and autonomous_mode:
                #     autonomous_mode = False
                #     mode = "MANUAL"
                #     status = f"Mode switched to {mode}"
                #     last_pressed_key = None
                #     print(status)
                #     continue

                # elif last_pressed_key == 'k':
                #     kill_switch_engaged = not kill_switch_engaged
                #     state = "ENGAGED" if kill_switch_engaged else "DISENGAGED"
                #     status = f"Kill switch {state}."
                #     last_pressed_key = None
                #     print(status)
                #     if not kill_switch_engaged:
                #         tello.send_rc_control(0, 0, 0, 0)
                #     else:
                #         tello.send_rc_control(0, 100, 0, 0)
                #     continue

                # elif last_pressed_key == 'c':
                #     circle_mode = not circle_mode
                #     mode = "CIRCLE MODE" if circle_mode else "NORMAL MODE"
                #     status = f"Circle mode {mode}."
                #     last_pressed_key = None
                #     print(status)
                #     continue

                # elif last_pressed_key == 'z':
                #     last_pressed_key = None
                #     timestamp = time.strftime("%Y%m%d-%H%M%S")
                #     save_dir = "./pics"
                #     filename = os.path.join(save_dir, f"snapshot_{timestamp}.jpg")
                #     snapshot = frame.copy()
                #     cv2.putText(snapshot, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                #                 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                #     cv2.putText(snapshot, current_time, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                #                 0.6, (0, 255, 255), 1, cv2.LINE_AA)
                #     cv2.imwrite(filename, snapshot)
                #     print(f"Snapshot saved as {filename}")
                #     continue
                
                
                with detected_box_lock:
                    if detected_box is None:
                        continue
                    det = detected_box.copy()

                # if kill_switch_engaged and det is None:
                #     status = "No bounding box detected. Disengaging kill switch."
                #     print(status)
                #     kill_switch_engaged = False
                #     tello.send_rc_control(0, 0, 0, 0)

                #     if autonomous_mode:
                #         status = "Rotating to find target..."
                #         print(status)
                #         tello.rotate_clockwise(90)
                #         time.sleep(1)
                    
                #     continue
                
                # if not autonomous_mode:
                #     if last_pressed_key == 'w':
                #         status = "Manual control: Moving forward 50 cm"
                #         last_pressed_key = None
                #         print(status)
                #         tello.move_forward(50)
                #     elif last_pressed_key == 'a':
                #         status = "Manual control: Moving left 50 cm"
                #         last_pressed_key = None
                #         print(status)
                #         tello.move_left(50)
                #     elif last_pressed_key == 's':
                #         status = "Manual control: Moving back 50 cm"
                #         last_pressed_key = None
                #         print(status)
                #         tello.move_back(50)
                #     elif last_pressed_key == 'd':
                #         status = "Manual control: Moving right 50 cm"
                #         last_pressed_key = None
                #         print(status)
                #         tello.move_right(50)
                #     elif last_pressed_key == 'u':
                #         status = "Manual control: Moving up 30 cm"
                #         last_pressed_key = None
                #         print(status)
                #         tello.move_up(30)
                #     elif last_pressed_key == 'j':
                #         status = "Manual control: Moving down 30 cm"
                #         last_pressed_key = None
                #         print(status)
                #         tello.move_down(30)
                #     elif last_pressed_key == 'e':
                #         status = "Manual control: Rotating clockwise 30°"
                #         last_pressed_key = None
                #         print(status)
                #         tello.rotate_clockwise(30)
                #     elif last_pressed_key == 'q':
                #         status = "Manual control: Rotating counter-clockwise 30°"
                #         last_pressed_key = None
                #         print(status)
                #         tello.rotate_counter_clockwise(30)

                # else:
                # if det is not None:
                #     class_id = int(det[1])
                #     class_name = class_names[class_id - 1]
                #     box_color = (0, 255, 0) if not kill_switch_engaged else (0, 0, 255)

                #     x1 = int(det[3] * w)
                #     y1 = int(det[4] * h)
                #     x2 = int(det[5] * w)
                #     y2 = int(det[6] * h)

                #     box_area = (x2 - x1) * (y2 - y1)
                #     image_area = w * h
                #     area_ratio = box_area / image_area
                #     area_error = area_ratio - area_target

                #     print("Area_ratio: ", area_ratio)
                #     bbox_center_x = (x1 + x2) / 2
                #     frame_center_x = w / 2
                #     x_offset = bbox_center_x - frame_center_x
                    
                #     # Activate circle mode
                #     if circle_mode:
                #         angle_for_circle = 360
                #         omega_for_circle = 36
                #         segments = int(360 / angle_for_circle)
                #         circle_image_counter = 1
                #         for _ in range(segments):
                #             radius_for_circle = conv_ratio_to_radius(area_ratio)
                #             print("radius_for_circle: ", radius_for_circle)
                #             #radius_for_circle = 50 #65 #50                        
                #             circle_image_counter = go_in_circle(angle_for_circle, omega_for_circle, radius_for_circle, tello, circle_image_counter)
                #             time.sleep(0.5)

                #         go_in_circle(tello, angle_for_circle, omega_for_circle, radius_for_circle)
                #         continue
                
                #     if drone_in_air and autonomous_mode:
                #         if kill_switch_engaged:
                #             status = "[KILL SWITCH ENGAGED] Full send forward"
                #             print(status)
                #             tello.send_rc_control(0, 100, 0, 0)
                #         else:
                #             tello.send_rc_control(0, 0, 0, 0)
                #             if (time.time() - last_move_time) > move_interval:
                #                 moved = False

                #                 if area_error < -area_tolerance:
                #                     status = f"Moving forward {move_step_cm} cm"
                #                     print(status)
                #                     tello.move_forward(move_step_cm)
                #                     moved = True
                #                 elif area_error > area_tolerance:
                #                     status = f"Moving back {move_step_cm} cm"
                #                     print(status)
                #                     tello.move_back(move_step_cm)
                #                     moved = True

                #                 if abs(x_offset) > pixel_threshold:
                #                     if x_offset > 0:
                #                         status = f"Moving right {move_step_cm} cm"
                #                         print(status)
                #                         tello.move_right(move_step_cm)
                #                     else:
                #                         status = f"Moving left {move_step_cm} cm"
                #                         print(status)
                #                         tello.move_left(move_step_cm)
                #                     moved = True

                #                 if not moved:
                #                     status = "Within tolerance. Hovering."
                #                     print(status)

                #                 print(status)

                #                 last_move_time = time.time()
                # else:
                #     status = "Rotating to find target..."
                #     print(status)
                #     tello.rotate_clockwise(90)
                #     time.sleep(1)
                
                # # Draw status text
                # cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                #             0.8, (0, 255, 0), 2, cv2.LINE_AA)

                # Draw time text just below the status
                # cv2.putText(frame, current_time, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                #             0.6, (0, 255, 255), 1, cv2.LINE_AA)

                # cv2.imshow("Tello Tracking Feed", frame)

            except Exception as e:
                print(f"Error in run_controller: {e}")
            print(f"[{threading.current_thread().name}] Running controller logic...")
        except Exception as e:
            print(f"[{threading.current_thread().name}] Error in run_controller: {e}")
        #time.sleep(0.033)  # ~30 FPS

#Large updates here
def generate_frames():
    global clicked_coordinates
    global model 
    global objects
    global drone_in_air
    global last_pressed_key
    global tello
    global detected_box
    global detected_box_lock
    global frame
    global frame_lock

    print(f"[{threading.current_thread().name}] Starting video frame generation...")
    while not tello:
        with frame_lock:
            frame = create_dummy_frame("Waiting for Tello connection")            
        # Convert frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        # Convert to base64 for WebSocket transmission
        frame_base64 = base64.b64encode(frame_bytes).decode('utf-8')
        socketio.emit('video_frame', {'frame': frame_base64})
        tello = get_tello()
        print(f"[{threading.current_thread().name}] Waiting for Tello connection...")

    first_frame = True
    selected = False
    # get off the ground

    print('last pressed key is ', last_pressed_key)
    while not drone_in_air:
        #print("Drone not in air")
        with frame_lock:
            frame = create_dummy_frame("Waiting for takeoff")            
        # Convert frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        # Convert to base64 for WebSocket transmission
        frame_base64 = base64.b64encode(frame_bytes).decode('utf-8')
        socketio.emit('video_frame', {'frame': frame_base64})

        #print(drone_in_air)
        if last_pressed_key == 't' and not drone_in_air:
            last_pressed_key = None
            print('okay')
            tello.takeoff()
            drone_in_air = True
            status = "Drone is now airborne."
            print(status)
            tello.move_up(50)
        print(f"[{threading.current_thread().name}] Waiting for drone takeoff...")

    frame_read = tello.get_frame_read()
    while True:
        with video_thread_lock:
            try:
                if tello is not None:
                    with frame_lock:
                        if frame_read is not None and frame_read.frame is not None:
                            frame = frame_read.frame
                            # Convert BGR to RGB
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            # Resize frame for better performance
                            frame = cv2.resize(frame, (320, 240))
                        else:
                            frame = create_dummy_frame("No Tello Feed Available")
                    
                    h, w, _ = frame.shape
                    # Object detection
                    blob = cv2.dnn.blobFromImage(frame, size=(300, 300), mean=(104, 117, 123), swapRB=True)
                    model.setInput(blob)
                    detections = model.forward()
                    #print(first_frame)
                    # if drone_in_air:
                    if first_frame: 
                        #print("enterd")
                        id_num = 1
                        # All detections need to be above threshold
                        #print(detections[0,0])
                        probs = [det[2] for det in detections[0,0]]
                        #print(probs)
                        if np.all(np.array(probs) <= 0.4):
                            #print('none')
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
                            #print('reached initial')
                            #print(len(objects))
                            first_frame = False
                    else:
                        #print('thru objects')
                        for obj in objects:
                            id_num = obj["id"]
                            old_det = obj["det"]
                            old_bbox = obj["bbox"]
                            new_det = closest_detection(detections, old_det)
                            if new_det is None:
                                print("new_det is None")
                                new_det = old_det
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
                    # Drawing bboxes 
                    for obj in objects:
                        #print(objects)
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
                        #print("Drawing object ", id_num)
                        #print('drawing frames')
                    
                    if selected is False:
                        # bounding box selection code
                        if clicked_coordinates["x"] is not None and clicked_coordinates["y"] is not None:
                            #print("checking coords")
                            objects_temp = []
                            for obj in objects:
                                if clicked_coordinates["x"] >= obj["bbox"]["x1"] and clicked_coordinates["x"] <= obj["bbox"]["x2"] and \
                                clicked_coordinates["y"] >= obj["bbox"]["y1"] and clicked_coordinates["y"] <= obj["bbox"]["y2"]:
                                    objects_temp.append(obj)
                            if len(objects_temp) > 0:
                                objects = objects_temp
                                print("Locking onto Person ", objects[0]["id"])
                                selected = True
                        #print("updated objects: ", objects)
                    else:
                        with detected_box_lock:
                            detected_box = objects[0]["det"]

                        # run_controller(det, drone_in_air, autonomous_mode, kill_switch_engaged, frame)
    
                # Convert frame to JPEG
                with frame_lock:
                    ret, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                # Convert to base64 for WebSocket transmission
                frame_base64 = base64.b64encode(frame_bytes).decode('utf-8')
                socketio.emit('video_frame', {'frame': frame_base64})
                print(f"[{threading.current_thread().name}] Processing video frame...")
            
            except KeyboardInterrupt:
                print("Interrupted. Landing drone.")
                if drone_in_air:
                    tello.land()

            except Exception as e:
                print(f"[{threading.current_thread().name}] Error in frame generation: {e}")
                with frame_lock:
                    frame = create_dummy_frame("Error in frame generation")
                ret, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                frame_base64 = base64.b64encode(frame_bytes).decode('utf-8')
                socketio.emit('video_frame', {'frame': frame_base64})

        #time.sleep(0.033)  # ~30 FPS

def start_video_thread():
    global video_thread
    global video_thread_lock
    with video_thread_lock:
        if video_thread is None:
            print(f"[{threading.current_thread().name}] Starting video thread...")
            video_thread = threading.Thread(target=generate_frames, name="VideoThread")
            video_thread.daemon = True
            video_thread.start()

def start_controller_thread():
    global controller_thread
    global controller_thread_lock
    with controller_thread_lock:
        if controller_thread is None:
            print(f"[{threading.current_thread().name}] Starting controller thread...")
            controller_thread = threading.Thread(target=run_controller, name="ControllerThread")
            controller_thread.daemon = True
            controller_thread.start()

start_video_thread()
start_controller_thread()

# @socketio.on('connect')
# def handle_connect():
#     print(f"[{threading.current_thread().name}] Client connected. Starting threads...")

@app.route('/')
def index():
    return "Tello Camera Stream Server"

@app.route('/coordinates', methods=['POST'])
def handle_coordinates():
    global clicked_coordinates
    try:
        data = request.get_json()
        clicked_coordinates["x"] = data.get('x')
        clicked_coordinates["y"] = data.get('y')
        print(f"Received coordinates: x={clicked_coordinates['x']}, y={clicked_coordinates['y']}")
        return jsonify({'message': 'Coordinates received successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/keypress', methods=['POST'])
def handle_keypress():
    global last_pressed_key
    try:
        data = request.get_json()
        key = data.get('key')
        if key:
            last_pressed_key = key
            print(f"Received key: {last_pressed_key}")
            return jsonify({'message': 'Key received successfully'}), 200
        else:
            return jsonify({'error': 'No key provided'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5001, debug=True)