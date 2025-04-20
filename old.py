from djitellopy import Tello
import cv2
import numpy as np
import time
import logging
import os


############
# Detection stuff
############
# Suppress djitellopy logs lower than WARNING
logging.getLogger('djitellopy').setLevel(logging.WARNING)

def detection_center(det):
    cx = (det[3] + det[5]) / 2.0 - 0.5
    cy = (det[4] + det[6]) / 2.0 - 0.5
    return (cx, cy)

def norm(vec):
    return np.sqrt(vec[0]**2 + vec[1]**2)

def closest_detection(detections):
    best_det = None
    min_dist = float('inf')
    for det in detections[0, 0]:
        if det[2] > 0.4 and int(det[1]) == 1:  # 'person'
            center = detection_center(det)
            distance = norm(center)
            if distance < min_dist:
                min_dist = distance
                best_det = det
    return best_det

with open('COCO/object_detection_classes_coco.txt', 'r') as f:
    class_names = f.read().split('\n')

COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))

model = cv2.dnn.readNet(model='COCO/frozen_inference_graph.pb',
                        config='COCO/ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt',
                        framework='TensorFlow')


############
# Circle geometry
############
def go_in_circle(angle, omega, radius):
    circle_time = angle/omega
    tello.send_rc_control(int(radius*np.deg2rad(angle)/(angle/omega)), 0, 0, int(-1*omega))
    #time.sleep(circle_time * 2)
    #tello.send_rc_control(0, 0, 0, 0)
    time.sleep(1)



tello = Tello()
print("Connecting to Tello...")
tello.connect()
print(f"Battery level: {tello.get_battery()}%")
tello.streamon()

print("Press 't' to takeoff, 'l' to land and quit.")
drone_in_air = False
kill_switch_engaged = False
autonomous_mode = True

area_target = 0.325 #0.25
area_tolerance = 0.075 #0.05
move_step_cm = 25 #30
pixel_threshold = 50 #40

last_move_time = time.time()
move_interval = 0.8

status = "Idle"

# Custom snapshot save directory
save_dir = "./pics"
os.makedirs(save_dir, exist_ok=True)

try:
    while True:
        frame = tello.get_frame_read().frame
        frame = cv2.resize(frame, (720, 480))
        h, w, _ = frame.shape

        blob = cv2.dnn.blobFromImage(frame, size=(300, 300), mean=(104, 117, 123), swapRB=True)
        model.setInput(blob)
        detections = model.forward()
        det = closest_detection(detections)

        if det is not None:
            class_id = int(det[1])
            class_name = class_names[class_id - 1]
            box_color = (0, 255, 0) if not kill_switch_engaged else (0, 0, 255)

            x1 = int(det[3] * w)
            y1 = int(det[4] * h)
            x2 = int(det[5] * w)
            y2 = int(det[6] * h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            label = f"{class_name} {det[2]:.2f}"
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

            box_area = (x2 - x1) * (y2 - y1)
            image_area = w * h
            area_ratio = box_area / image_area
            area_error = area_ratio - area_target

            bbox_center_x = (x1 + x2) / 2
            frame_center_x = w / 2
            x_offset = bbox_center_x - frame_center_x

            if drone_in_air and autonomous_mode:
                if kill_switch_engaged:
                    status = "[KILL SWITCH ENGAGED] Full send forward"
                    print(status)
                    tello.send_rc_control(0, 100, 0, 0)
                else:
                    tello.send_rc_control(0, 0, 0, 0)
                    if (time.time() - last_move_time) > move_interval:
                        moved = False

                        if area_error < -area_tolerance:
                            status = f"Moving forward {move_step_cm} cm"
                            print(status)
                            tello.move_forward(move_step_cm)
                            moved = True
                        elif area_error > area_tolerance:
                            status = f"Moving back {move_step_cm} cm"
                            print(status)
                            tello.move_back(move_step_cm)
                            moved = True

                        if abs(x_offset) > pixel_threshold:
                            if x_offset > 0:
                                status = f"Moving right {move_step_cm} cm"
                                print(status)
                                tello.move_right(move_step_cm)
                            else:
                                status = f"Moving left {move_step_cm} cm"
                                print(status)
                                tello.move_left(move_step_cm)
                            moved = True

                        if not moved:
                            status = "Within tolerance. Hovering."
                            print(status)

                        last_move_time = time.time()

        if det is None and kill_switch_engaged:
            status = "No bounding box detected. Disengaging kill switch."
            print(status)
            kill_switch_engaged = False
            tello.send_rc_control(0, 0, 0, 0)

            if autonomous_mode:
                status = "Rotating to find target..."
                print(status)
                tello.rotate_clockwise(90)
                time.sleep(1)

        # Draw status text
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2, cv2.LINE_AA)

        # Get current time string
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")

        # Draw time text just below the status
        cv2.putText(frame, current_time, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Tello Tracking Feed", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('t') and not drone_in_air:
            tello.takeoff()
            drone_in_air = True
            status = "Drone is now airborne."
            print(status)
            tello.move_up(50)

        if drone_in_air:
            if key == ord('w'):
                status = "Manual control: Moving forward 50 cm"
                print(status)
                tello.move_forward(50)
            elif key == ord('a'):
                status = "Manual control: Moving left 50 cm"
                print(status)
                tello.move_left(50)
            elif key == ord('s'):
                status = "Manual control: Moving back 50 cm"
                print(status)
                tello.move_back(50)
            elif key == ord('d'):
                status = "Manual control: Moving right 50 cm"
                print(status)
                tello.move_right(50)
            elif key == ord('u'):
                status = "Manual control: Moving up 30 cm"
                print(status)
                tello.move_up(30)
            elif key == ord('j'):
                status = "Manual control: Moving down 30 cm"
                print(status)
                tello.move_down(30)
            elif key == ord('e'):
                status = "Manual control: Rotating clockwise 30°"
                print(status)
                tello.rotate_clockwise(30)
            elif key == ord('q'):
                status = "Manual control: Rotating counter-clockwise 30°"
                print(status)
                tello.rotate_counter_clockwise(30)
            elif key == ord('c'):
                status = "going in circle"
                print(status)
                angle_for_circle = 360
                omega_for_circle = 36
                radius_for_circle = 50 #65 #50
                go_in_circle(angle_for_circle, omega_for_circle, radius_for_circle)

        if key == ord('1'):
            autonomous_mode = not autonomous_mode
            mode = "AUTONOMOUS" if autonomous_mode else "MANUAL"
            status = f"Mode switched to {mode}"
            print(status)

        elif key == ord('k'):
            kill_switch_engaged = not kill_switch_engaged
            state = "ENGAGED" if kill_switch_engaged else "DISENGAGED"
            status = f"Kill switch {state}."
            print(status)
            if not kill_switch_engaged:
                tello.send_rc_control(0, 0, 0, 0)

        elif key == ord('z'):
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = os.path.join(save_dir, f"snapshot_{timestamp}.jpg")
            snapshot = frame.copy()
            cv2.putText(snapshot, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(snapshot, current_time, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.imwrite(filename, snapshot)
            print(f"Snapshot saved as {filename}")

        elif key == ord('l'):
            if drone_in_air:
                tello.land()
                status = "Drone has landed."
                print(status)
            break

except Exception as e:
    print(f"Error: {e}")
    if drone_in_air:
        tello.land()

except KeyboardInterrupt:
    print("Interrupted. Landing drone.")
    if drone_in_air:
        tello.land()

finally:
    tello.streamoff()
    cv2.destroyAllWindows()
