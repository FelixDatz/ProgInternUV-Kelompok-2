import cv2
from ultralytics import YOLO
import numpy as np
import time
import collections

try:
    from collections import abc
    collections.MutableMapping = abc.MutableMapping
except Exception:
    pass

from dronekit import connect, VehicleMode, Command
from pymavlink import mavutil

# ==============================
# LOAD YOLO MODEL
# ==============================

model = YOLO("best.pt")

# ==============================
# CAMERA SETUP
# ==============================

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # ← batasi resolusi kamera
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# ==============================
# PARAMETERS
# ==============================

REAL_BALL_DIAMETER     = 20.0
FOCAL_LENGTH           = 700
FRAME_CENTER_TOLERANCE = 40

OBSTACLE_DISTANCE_THRESHOLD = 80
ORBIT_RIGHT_ZONE     = 0.60
ORBIT_KEEP_MIN       = 0.50
ORBIT_KEEP_MAX       = 0.85

PWM_LEFT   = 1400
PWM_CENTER = 1500
PWM_RIGHT  = 1600

NO_DETECTION_TIMEOUT       = 30.0
NO_DETECTION_ORBIT_TIMEOUT = 10.0
ORBIT_CONFIRM_NEEDED       = 5

# ← BARU: ukuran input YOLO (lebih kecil = lebih cepat)
YOLO_IMGSZ = 320

# ← BARU: jalankan YOLO tiap N frame (skip frame)
YOLO_SKIP_FRAMES = 2   # 1 = tiap frame, 2 = selang-seling, 3 = tiap 3 frame

# ==============================
# MAVLINK VELOCITY PARAMETERS
# ==============================

SPEED_FORWARD = 2.0
SPEED_TURN    = 1.0
SPEED_STOP    = 0.0

# ==============================
# STATE MACHINE
# ==============================

STATE_GATE       = "GATE"
STATE_CIRC_START = "CIRC_START"
STATE_CIRC_ORBIT = "CIRC_ORBIT"
STATE_SHUTDOWN   = "SHUTDOWN"

nav_state = STATE_GATE

# ==============================
# INTERNAL COUNTERS
# ==============================

last_detection_time = time.time()
orbit_confirm_count = 0
frame_count         = 0          # ← BARU: untuk frame skip

# Cache hasil deteksi terakhir (dipakai saat frame di-skip)
cached_boxes = []                 # ← BARU

# ==============================
# MAVLINK CONNECTION
# ==============================

print("Connecting to vehicle...")
vehicle = connect('tcp:127.0.0.1:5762')
print("Connected!")

# ==============================
# MAVLINK HELPERS
# ==============================

def add_last_waypoint_to_mission(wp_lat, wp_lon, wp_alt):
    cmds = vehicle.commands
    cmds.download()
    cmds.wait_ready()
    mission_list = [cmd for cmd in cmds]
    wp_obj = Command(
        0, 0, 0,
        mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
        mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
        0, 0, 0, 0, 0, 0,
        wp_lat, wp_lon, wp_alt
    )
    mission_list.append(wp_obj)
    cmds.clear()
    for cmd in mission_list:
        cmds.add(cmd)
    cmds.upload()
    return cmds.count


def gerak(vx, vy):
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0, 0, 0,
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
        0b0000111111000111,
        0, 0, 0,
        vx, vy, 0,
        0, 0, 0,
        0, 0
    )
    vehicle.send_mavlink(msg)
    vehicle.flush()


def gerak_stop():  gerak(SPEED_STOP, SPEED_STOP)
def gerak_maju():  gerak(SPEED_FORWARD, 0)
def gerak_kiri():  gerak(SPEED_FORWARD, -SPEED_TURN)
def gerak_kanan(): gerak(SPEED_FORWARD, SPEED_TURN)

def set_steering(pwm): print(f"STEER PWM: {pwm}")
def throttle_forward(): print("THROTTLE: forward")
def throttle_stop():    print("THROTTLE: stop")

# ==============================
# DISTANCE ESTIMATION
# ==============================

def estimate_distance(box):
    x1, y1, x2, y2 = box
    diameter = max(x2 - x1, y2 - y1)
    if diameter == 0:
        return None
    return (REAL_BALL_DIAMETER * FOCAL_LENGTH) / diameter


# ==============================
# BARU: Gambar box secara manual (pengganti results[0].plot())
# ==============================

COLOR_MAP = {
    "bola_merah": (0, 0, 255),
    "bola_hijau": (0, 255, 0),
    "bola_hitam": (80, 80, 80),
}

def draw_boxes_manual(frame, boxes_data):
    """
    boxes_data: list of (x1,y1,x2,y2, name, distance)
    Jauh lebih cepat dari results[0].plot()
    """
    for (x1, y1, x2, y2, name, distance) in boxes_data:
        color = COLOR_MAP.get(name, (200, 200, 200))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{name} {int(distance)}cm"
        cv2.putText(frame, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return frame


# ==============================
# INIT
# ==============================

add_last_waypoint_to_mission(
    vehicle.location.global_relative_frame.lat,
    vehicle.location.global_relative_frame.lon,
    vehicle.location.global_relative_frame.alt
)
print("Home waypoint added to mission")

vehicle.mode  = VehicleMode("GUIDED")
vehicle.armed = True
print("Vehicle armed in GUIDED mode")

# ==============================
# MAIN LOOP
# ==============================

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera error")
        break

    frame_count += 1
    frame_height, frame_width = frame.shape[:2]
    frame_center = frame_width / 2

    # ——————————————————————————————————————
    # YOLO hanya dijalankan tiap YOLO_SKIP_FRAMES frame
    # ——————————————————————————————————————
    if frame_count % YOLO_SKIP_FRAMES == 0:
        # Resize frame kecil untuk inferensi
        small = cv2.resize(frame, (YOLO_IMGSZ, YOLO_IMGSZ))
        results = model(small, imgsz=YOLO_IMGSZ, verbose=False)

        scale_x = frame_width  / YOLO_IMGSZ
        scale_y = frame_height / YOLO_IMGSZ

        boxes_data = []
        for box in results[0].boxes:
            cls  = int(box.cls)
            name = model.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Skala balik ke koordinat frame asli
            x1 = int(x1 * scale_x); x2 = int(x2 * scale_x)
            y1 = int(y1 * scale_y); y2 = int(y2 * scale_y)

            distance = estimate_distance((x1, y1, x2, y2))
            if distance is None:
                continue
            boxes_data.append((x1, y1, x2, y2, name, distance))

        cached_boxes = boxes_data   # simpan cache untuk frame yang di-skip

    else:
        boxes_data = cached_boxes   # pakai hasil deteksi sebelumnya

    # ——————————————————————————————————————
    # PROSES DETEKSI dari boxes_data
    # ——————————————————————————————————————
    red_x          = None; red_distance   = 9999
    green_x        = None; green_distance = 9999
    black_x        = None; black_distance = 9999

    for (x1, y1, x2, y2, name, distance) in boxes_data:
        x_center = (x1 + x2) / 2

        if name == "bola_merah":
            if distance < red_distance:
                red_x = x_center; red_distance = distance
        elif name == "bola_hijau":
            if distance < green_distance:
                green_x = x_center; green_distance = distance
        elif name == "bola_hitam":
            if distance < black_distance:
                black_x = x_center; black_distance = distance

    # ——————————————————————————————————————
    # UPDATE DETECTION TIMER
    # ——————————————————————————————————————
    any_detected = (red_x is not None or green_x is not None or black_x is not None)
    if any_detected:
        last_detection_time = time.time()
    time_since_detection = time.time() - last_detection_time

    # ——————————————————————————————————————
    # STATE TRANSITION (tidak berubah)
    # ——————————————————————————————————————
    gate_visible  = (red_x is not None and green_x is not None)
    obstacle_near = (black_x is not None and black_distance < OBSTACLE_DISTANCE_THRESHOLD)
    only_black    = (black_x is not None and red_x is None and green_x is None)
    black_ratio   = (black_x / frame_width) if black_x is not None else None

    timeout_limit = NO_DETECTION_ORBIT_TIMEOUT if nav_state == STATE_CIRC_ORBIT else NO_DETECTION_TIMEOUT

    if time_since_detection >= timeout_limit and nav_state != STATE_SHUTDOWN:
        nav_state = STATE_SHUTDOWN
        print(f">> STATE: SHUTDOWN (no detection for {time_since_detection:.1f}s)")

    elif nav_state == STATE_GATE:
        if (obstacle_near or only_black) and not gate_visible:
            orbit_confirm_count = 0
            nav_state = STATE_CIRC_START
            print(">> STATE: CIRC_START")

    elif nav_state == STATE_CIRC_START:
        if black_ratio is not None and black_ratio >= ORBIT_RIGHT_ZONE:
            orbit_confirm_count += 1
            if orbit_confirm_count >= ORBIT_CONFIRM_NEEDED:
                nav_state = STATE_CIRC_ORBIT
                orbit_confirm_count = 0
                print(">> STATE: CIRC_ORBIT")
        else:
            orbit_confirm_count = 0
        if gate_visible:
            orbit_confirm_count = 0
            nav_state = STATE_GATE
            print(">> STATE: GATE (reacquired)")

    elif nav_state == STATE_CIRC_ORBIT:
        if gate_visible:
            nav_state = STATE_GATE
            print(">> STATE: GATE (after orbit)")
        elif black_x is None:
            nav_state = STATE_GATE
            print(">> STATE: GATE (black lost)")

    # ——————————————————————————————————————
    # NAVIGATION DECISION + MAVLINK
    # ——————————————————————————————————————
    steering = PWM_CENTER

    if nav_state == STATE_GATE:
        if gate_visible:
            path_center = (red_x + green_x) / 2
            if path_center < frame_center - FRAME_CENTER_TOLERANCE:
                steering = PWM_LEFT;   gerak_kiri()
            elif path_center > frame_center + FRAME_CENTER_TOLERANCE:
                steering = PWM_RIGHT;  gerak_kanan()
            else:
                steering = PWM_CENTER; gerak_maju()
        elif red_x is not None:
            steering = PWM_RIGHT; gerak_kanan()
        elif green_x is not None:
            steering = PWM_LEFT;  gerak_kiri()
        else:
            gerak_maju()

    elif nav_state == STATE_CIRC_START:
        steering = PWM_LEFT; gerak_kiri()

    elif nav_state == STATE_CIRC_ORBIT:
        if black_ratio is not None:
            if black_ratio < ORBIT_KEEP_MIN:
                steering = PWM_LEFT;   gerak_kiri()
            elif black_ratio > ORBIT_KEEP_MAX:
                steering = PWM_RIGHT;  gerak_kanan()
            else:
                steering = PWM_RIGHT;  gerak_kanan()
        else:
            steering = PWM_RIGHT; gerak_kanan()

    elif nav_state == STATE_SHUTDOWN:
        gerak_stop()
        throttle_stop()

    # ——————————————————————————————————————
    # VISUALISASI — gambar manual, tanpa .plot()
    # ——————————————————————————————————————
    annotated = draw_boxes_manual(frame, boxes_data)

    cv2.line(annotated, (int(frame_center), 0), (int(frame_center), frame_height), (255, 255, 0), 1)

    for ratio, color in [
        (ORBIT_KEEP_MIN,   (0, 200,   0)),
        (ORBIT_KEEP_MAX,   (0,   0, 200)),
        (ORBIT_RIGHT_ZONE, (0, 165, 255)),
    ]:
        cx = int(frame_width * ratio)
        cv2.line(annotated, (cx, 0), (cx, frame_height), color, 1)

    color_map = {
        STATE_GATE       : (0, 255, 255),
        STATE_CIRC_START : (0, 165, 255),
        STATE_CIRC_ORBIT : (0, 255,   0),
        STATE_SHUTDOWN   : (0,   0, 255),
    }
    cv2.putText(annotated, f"STATE: {nav_state}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_map.get(nav_state, (255,255,255)), 2, cv2.LINE_AA)

    if gate_visible:
        cv2.putText(annotated, f"Gate: R={int(red_distance)}cm G={int(green_distance)}cm",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

    if black_ratio is not None:
        cv2.putText(annotated, f"Black X:{black_ratio:.2f} {int(black_distance)}cm",
                    (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    if time_since_detection > 1.0 and nav_state != STATE_SHUTDOWN:
        remaining = timeout_limit - time_since_detection
        cv2.putText(annotated, f"No det: {remaining:.1f}s",
                    (10, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 50, 255), 1, cv2.LINE_AA)

    if nav_state == STATE_CIRC_START and orbit_confirm_count > 0:
        cv2.putText(annotated, f"Confirm: {orbit_confirm_count}/{ORBIT_CONFIRM_NEEDED}",
                    (10, 116), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1, cv2.LINE_AA)

    if only_black and nav_state in (STATE_CIRC_START, STATE_CIRC_ORBIT):
        cv2.putText(annotated, "ONLY BLACK — orbit mode",
                    (10, 138), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 1, cv2.LINE_AA)

    if nav_state == STATE_SHUTDOWN:
        cv2.putText(annotated, "!! SHUTDOWN !!",
                    (int(frame_width * 0.2), frame_height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)

    cv2.imshow("Naval Cam", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        gerak_stop()
        break

    if nav_state == STATE_SHUTDOWN:
        cv2.waitKey(1000)
        break

# ==============================
# CLEANUP
# ==============================

gerak_stop()
cap.release()
cv2.destroyAllWindows()
vehicle.close()
print("Program selesai.")
