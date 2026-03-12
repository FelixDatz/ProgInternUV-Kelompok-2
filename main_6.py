import cv2
from ultralytics import YOLO
import numpy as np
import time
import collections

# ==============================
# DRONEKIT / MAVLINK SETUP
# ==============================

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

cap = cv2.VideoCapture(0)

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

# PWM steering values (untuk logging / referensi)
PWM_LEFT   = 1400
PWM_CENTER = 1500
PWM_RIGHT  = 1600

NO_DETECTION_TIMEOUT       = 5.0
NO_DETECTION_ORBIT_TIMEOUT = 10.0
ORBIT_CONFIRM_NEEDED       = 5

# ==============================
# MAVLINK VELOCITY PARAMETERS
# ==============================

SPEED_FORWARD     = 2.0    # m/s  — kecepatan maju normal
SPEED_TURN        = 1.0    # m/s  — kecepatan lateral saat belok
SPEED_STOP        = 0.0    # m/s

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
    """Tambahkan satu waypoint terakhir ke misi aktif."""
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
    """
    Kirim perintah velocity ke kapal.
    vx = maju/mundur  (m/s, positif = maju)
    vy = kanan/kiri   (m/s, positif = kanan, negatif = kiri)
    Frame: MAV_FRAME_BODY_OFFSET_NED (relatif terhadap orientasi kapal)
    """
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,
        0, 0,
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
        0b0000111111000111,   # mask: hanya vx, vy, vz aktif
        0, 0, 0,              # posisi (diabaikan)
        vx, vy, 0,            # velocity
        0, 0, 0,              # akselerasi (diabaikan)
        0, 0                  # yaw, yaw_rate (diabaikan)
    )
    vehicle.send_mavlink(msg)
    vehicle.flush()


def gerak_stop():
    """Hentikan kapal (velocity = 0)."""
    gerak(SPEED_STOP, SPEED_STOP)


def gerak_maju():
    """Maju lurus."""
    gerak(SPEED_FORWARD, 0)


def gerak_kiri():
    """Maju sambil belok kiri."""
    gerak(SPEED_FORWARD, -SPEED_TURN)


def gerak_kanan():
    """Maju sambil belok kanan."""
    gerak(SPEED_FORWARD, SPEED_TURN)

# ==============================
# AUTOPILOT CONTROL (PWM logging)
# ==============================

def set_steering(pwm):
    """Log steering PWM (referensi visual di HUD)."""
    print(f"STEER PWM: {pwm}")


def throttle_forward():
    print("THROTTLE: forward")


def throttle_stop():
    print("THROTTLE: stop")

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
# INIT: HOME WAYPOINT + ARM
# ==============================

add_last_waypoint_to_mission(
    vehicle.location.global_relative_frame.lat,
    vehicle.location.global_relative_frame.lon,
    vehicle.location.global_relative_frame.alt
)
print("Home waypoint added to mission")

# Set GUIDED mode agar bisa terima velocity command
vehicle.mode = VehicleMode("GUIDED")
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

    frame_height, frame_width = frame.shape[:2]
    frame_center = frame_width / 2

    results = model(frame)
    annotated = results[0].plot()
    boxes = results[0].boxes

    # Reset detections
    red_x          = None
    red_distance   = 9999
    green_x        = None
    green_distance = 9999
    black_x        = None
    black_distance = 9999

    # ==============================
    # PROCESS DETECTIONS
    # ==============================

    for box in boxes:
        cls  = int(box.cls)
        name = model.names[cls]

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x_center = (x1 + x2) / 2

        distance = estimate_distance((x1, y1, x2, y2))
        if distance is None:
            continue

        cv2.putText(annotated, f"{int(distance)}cm",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)

        if name == "bola_merah":
            if distance < red_distance:
                red_x        = x_center
                red_distance = distance

        elif name == "bola_hijau":
            if distance < green_distance:
                green_x        = x_center
                green_distance = distance

        elif name == "bola_hitam":
            if distance < black_distance:
                black_x        = x_center
                black_distance = distance

    # ==============================
    # UPDATE DETECTION TIMER
    # ==============================

    any_detected = (red_x is not None or green_x is not None or black_x is not None)

    if any_detected:
        last_detection_time = time.time()

    time_since_detection = time.time() - last_detection_time

    # ==============================
    # STATE TRANSITION
    # ==============================

    gate_visible  = (red_x is not None and green_x is not None)
    obstacle_near = (black_x is not None and black_distance < OBSTACLE_DISTANCE_THRESHOLD)

    # [BARU] True jika HANYA black yang terdeteksi, tanpa red/green sama sekali
    only_black    = (black_x is not None and red_x is None and green_x is None)

    black_ratio   = (black_x / frame_width) if black_x is not None else None

    timeout_limit = NO_DETECTION_ORBIT_TIMEOUT if nav_state == STATE_CIRC_ORBIT else NO_DETECTION_TIMEOUT

    if time_since_detection >= timeout_limit and nav_state != STATE_SHUTDOWN:
        nav_state = STATE_SHUTDOWN
        print(f">> STATE: SHUTDOWN (no detection for {time_since_detection:.1f}s)")

    elif nav_state == STATE_GATE:
        # Masuk CIRC_START jika:
        #   1. Black terlalu dekat (obstacle_near), ATAU
        #   2. Hanya black yang terdeteksi tanpa red/green (only_black)
        # Kedua kondisi tidak memerlukan gate visible
        if (obstacle_near or only_black) and not gate_visible:
            orbit_confirm_count = 0
            nav_state = STATE_CIRC_START
            if only_black and not obstacle_near:
                print(">> STATE: CIRC_START (only black detected, no gate — no distance check)")
            else:
                print(">> STATE: CIRC_START (obstacle too close, turning left)")

    elif nav_state == STATE_CIRC_START:
        if black_ratio is not None and black_ratio >= ORBIT_RIGHT_ZONE:
            orbit_confirm_count += 1
            if orbit_confirm_count >= ORBIT_CONFIRM_NEEDED:
                nav_state = STATE_CIRC_ORBIT
                orbit_confirm_count = 0
                print(">> STATE: CIRC_ORBIT (black confirmed on right, orbiting)")
        else:
            orbit_confirm_count = 0

        if gate_visible:
            orbit_confirm_count = 0
            nav_state = STATE_GATE
            print(">> STATE: GATE (gate reacquired)")

    elif nav_state == STATE_CIRC_ORBIT:
        if gate_visible:
            nav_state = STATE_GATE
            print(">> STATE: GATE (gate reacquired after orbit)")
        elif black_x is None:
            nav_state = STATE_GATE
            print(">> STATE: GATE (black ball lost, resuming gate search)")

    # ==============================
    # NAVIGATION DECISION + MAVLINK
    # ==============================

    steering = PWM_CENTER   # untuk HUD log saja

    if nav_state == STATE_GATE:

        if gate_visible:
            path_center = (red_x + green_x) / 2
            if path_center < frame_center - FRAME_CENTER_TOLERANCE:
                print(f"Gate → Go LEFT  (mid={int(path_center)}, red={int(red_distance)}cm, green={int(green_distance)}cm)")
                steering = PWM_LEFT
                gerak_kiri()

            elif path_center > frame_center + FRAME_CENTER_TOLERANCE:
                print(f"Gate → Go RIGHT (mid={int(path_center)}, red={int(red_distance)}cm, green={int(green_distance)}cm)")
                steering = PWM_RIGHT
                gerak_kanan()

            else:
                print(f"Gate → STRAIGHT (mid={int(path_center)}, red={int(red_distance)}cm, green={int(green_distance)}cm)")
                steering = PWM_CENTER
                gerak_maju()

        elif red_x is not None:
            print(f"Only Red ({int(red_distance)}cm) → turn RIGHT")
            steering = PWM_RIGHT
            gerak_kanan()

        elif green_x is not None:
            print(f"Only Green ({int(green_distance)}cm) → turn LEFT")
            steering = PWM_LEFT
            gerak_kiri()

        else:
            print("Searching gate...")
            steering = PWM_CENTER
            gerak_maju()   # tetap maju pelan sambil scan

    elif nav_state == STATE_CIRC_START:
        label = f"black @ {black_ratio:.2f}" if black_ratio else "no black"
        print(f"Circ Start → Steer LEFT ({label})")
        steering = PWM_LEFT
        gerak_kiri()

    elif nav_state == STATE_CIRC_ORBIT:
        if black_ratio is not None:
            if black_ratio < ORBIT_KEEP_MIN:
                print(f"Orbit: drift left ({black_ratio:.2f}) → adjust LEFT")
                steering = PWM_LEFT
                gerak_kiri()

            elif black_ratio > ORBIT_KEEP_MAX:
                print(f"Orbit: too far right ({black_ratio:.2f}) → STRAIGHT")
                steering = PWM_CENTER
                gerak_maju()

            else:
                print(f"Orbit: orbiting ({black_ratio:.2f}) → Steer RIGHT")
                steering = PWM_RIGHT
                gerak_kanan()
        else:
            print("Orbit: black lost momentarily → Steer RIGHT")
            steering = PWM_RIGHT
            gerak_kanan()

    elif nav_state == STATE_SHUTDOWN:
        gerak_stop()
        set_steering(PWM_CENTER)
        throttle_stop()

    # ==============================
    # VISUALIZATION
    # ==============================

    cv2.line(annotated,
             (int(frame_center), 0), (int(frame_center), frame_height),
             (255, 255, 0), 2)

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
    cv2.putText(annotated, f"STATE: {nav_state}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, color_map.get(nav_state, (255, 255, 255)), 2)

    if gate_visible:
        cv2.putText(annotated,
                    f"Gate: red={int(red_distance)}cm  green={int(green_distance)}cm",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if black_ratio is not None:
        cv2.putText(annotated,
                    f"Black X: {black_ratio:.2f}  Dist: {int(black_distance)}cm",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    # [BARU] Label khusus jika only_black aktif
    if only_black and nav_state in (STATE_CIRC_START, STATE_CIRC_ORBIT):
        cv2.putText(annotated, "ONLY BLACK — orbit mode",
                    (10, 175), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 165, 255), 2)

    if time_since_detection > 1.0 and nav_state != STATE_SHUTDOWN:
        remaining = timeout_limit - time_since_detection
        cv2.putText(annotated, f"No detection: {remaining:.1f}s",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 50, 255), 2)

    if nav_state == STATE_CIRC_START and orbit_confirm_count > 0:
        cv2.putText(annotated,
                    f"Orbit confirm: {orbit_confirm_count}/{ORBIT_CONFIRM_NEEDED}",
                    (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

    if nav_state == STATE_SHUTDOWN:
        cv2.putText(annotated, "!! SHUTDOWN: NO DETECTION !!",
                    (int(frame_width * 0.05), frame_height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    cv2.imshow("Naval Cam", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        gerak_stop()
        break

    if nav_state == STATE_SHUTDOWN:
        cv2.waitKey(1500)
        break

# ==============================
# CLEANUP
# ==============================

gerak_stop()
cap.release()
cv2.destroyAllWindows()
vehicle.close()
print("Program selesai.")