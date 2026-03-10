import cv2
from ultralytics import YOLO
import numpy as np

# ==============================
# LOAD YOLO MODEL
# ==============================

model = YOLO("best.pt")

# ==============================
# CAMERA SETUP
# ==============================

cap = cv2.VideoCapture(2)
# cap = cv2.VideoCapture("http://192.168.0.107:8080/video")

# ==============================
# PARAMETERS
# ==============================

REAL_BALL_DIAMETER   = 20.0
FOCAL_LENGTH         = 700
FRAME_CENTER_TOLERANCE = 40

OBSTACLE_DISTANCE_THRESHOLD = 80    # cm, mulai circumnavigate
ORBIT_RIGHT_ZONE     = 0.60         # bola hitam dianggap "sudah di kanan" jika ratio > ini
ORBIT_KEEP_MIN       = 0.50         # batas minimum agar bola tetap terlihat di kanan
ORBIT_KEEP_MAX       = 0.85         # batas maximum, jangan sampai terlalu ke tepi

PWM_LEFT   = 1400
PWM_CENTER = 1500
PWM_RIGHT  = 1600

# ==============================
# STATE MACHINE
# ==============================

STATE_GATE           = "GATE"
STATE_CIRC_START     = "CIRC_START"   # belok kiri dulu, dorong hitam ke kanan
STATE_CIRC_ORBIT     = "CIRC_ORBIT"   # hitam sudah di kanan, orbit memutarinya

nav_state = STATE_GATE

# ==============================
# AUTOPILOT CONTROL (dummy)
# ==============================

def set_steering(pwm):
    print("STEER:", pwm)

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
    green_x        = None
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
            red_x = x_center
        elif name == "bola_hijau":
            green_x = x_center
        elif name == "bola_hitam":
            if distance < black_distance:
                black_distance = distance
                black_x = x_center

    # ==============================
    # STATE TRANSITION
    # ==============================

    gate_visible     = (red_x is not None and green_x is not None)
    obstacle_near    = (black_x is not None and black_distance < OBSTACLE_DISTANCE_THRESHOLD)
    black_ratio      = (black_x / frame_width) if black_x is not None else None

    if nav_state == STATE_GATE:
        # Trigger circumnavigate jika hitam dekat & gate tidak lengkap
        if obstacle_near and not gate_visible:
            nav_state = STATE_CIRC_START
            print(">> STATE: CIRC_START (obstacle detected on left, turning left)")

    elif nav_state == STATE_CIRC_START:
        # Transisi ke ORBIT ketika hitam sudah bergeser ke kanan frame
        if black_ratio is not None and black_ratio >= ORBIT_RIGHT_ZONE:
            nav_state = STATE_CIRC_ORBIT
            print(">> STATE: CIRC_ORBIT (black ball now on right, orbiting)")
        # Kembali ke GATE jika gate terlihat lagi
        elif gate_visible:
            nav_state = STATE_GATE
            print(">> STATE: GATE (gate reacquired)")

    elif nav_state == STATE_CIRC_ORBIT:
        # Kembali ke GATE jika gate terlihat / hitam sudah tidak terdeteksi
        if gate_visible:
            nav_state = STATE_GATE
            print(">> STATE: GATE (gate reacquired after orbit)")
        elif black_x is None:
            # Hitam hilang dari frame saat orbit → lanjut cari gate
            nav_state = STATE_GATE
            print(">> STATE: GATE (black ball lost, resuming gate search)")

    # ==============================
    # NAVIGATION DECISION
    # ==============================

    steering = PWM_CENTER

    # --------------------------------------------------
    # STATE: GATE  →  navigasi normal merah-hijau
    # --------------------------------------------------
    if nav_state == STATE_GATE:

        if gate_visible:
            path_center = (red_x + green_x) / 2

            if path_center > frame_center - FRAME_CENTER_TOLERANCE:
                print("Gate → Go LEFT")
                steering = PWM_LEFT
            elif path_center < frame_center + FRAME_CENTER_TOLERANCE:
                print("Gate → Go RIGHT")
                steering = PWM_RIGHT
            else:
                print("Gate → Go STRAIGHT")
                steering = PWM_CENTER

        elif red_x is not None:
            print("Only Red → turn RIGHT")
            steering = PWM_RIGHT
        elif green_x is not None:
            print("Only Green → turn LEFT")
            steering = PWM_LEFT
        else:
            print("Searching gate...")
            steering = PWM_CENTER

    # --------------------------------------------------
    # STATE: CIRC_START  →  belok KIRI sampai hitam ke kanan
    # --------------------------------------------------
    elif nav_state == STATE_CIRC_START:
        print(f"Circ Start → Steer LEFT (black @ {black_ratio:.2f})" if black_ratio else "Circ Start → Steer LEFT")
        steering = PWM_LEFT

    # --------------------------------------------------
    # STATE: CIRC_ORBIT  →  belok KANAN, jaga hitam di zona kanan
    # --------------------------------------------------
    elif nav_state == STATE_CIRC_ORBIT:

        if black_ratio is not None:
            if black_ratio < ORBIT_KEEP_MIN:
                # Hitam mulai keluar ke kiri → belok kiri sedikit untuk reacquire
                print(f"Orbit: black drifting left ({black_ratio:.2f}) → adjust LEFT")
                steering = PWM_LEFT
            elif black_ratio > ORBIT_KEEP_MAX:
                # Terlalu ke tepi kanan → sedikit lurus
                print(f"Orbit: black too far right ({black_ratio:.2f}) → STRAIGHT")
                steering = PWM_CENTER
            else:
                # Zona ideal → terus belok kanan memutari
                print(f"Orbit: orbiting right ({black_ratio:.2f}) → Steer RIGHT")
                steering = PWM_RIGHT
        else:
            # Kehilangan target saat orbit
            print("Orbit: black lost → Steer RIGHT to find")
            steering = PWM_RIGHT

    # ==============================
    # CONTROL OUTPUT
    # ==============================

    set_steering(steering)
    throttle_forward()

    # ==============================
    # VISUALIZATION
    # ==============================

    # Garis tengah
    cv2.line(annotated,
             (int(frame_center), 0), (int(frame_center), frame_height),
             (255, 255, 0), 2)

    # Zona orbit (garis batas kiri & kanan zona ideal)
    for ratio, color in [(ORBIT_KEEP_MIN, (0,200,0)), (ORBIT_KEEP_MAX, (0,0,200)), (ORBIT_RIGHT_ZONE, (0,165,255))]:
        cx = int(frame_width * ratio)
        cv2.line(annotated, (cx, 0), (cx, frame_height), color, 1)

    # HUD state
    color_map = {
        STATE_GATE       : (0, 255, 255),
        STATE_CIRC_START : (0, 165, 255),
        STATE_CIRC_ORBIT : (0, 255, 0),
    }
    cv2.putText(annotated, f"STATE: {nav_state}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, color_map.get(nav_state, (255,255,255)), 2)

    if black_ratio is not None:
        cv2.putText(annotated, f"Black X: {black_ratio:.2f}  Dist: {int(black_distance)}cm",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (200, 200, 200), 2)

    cv2.imshow("Naval Cam", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==============================
# CLEANUP
# ==============================

cap.release()
cv2.destroyAllWindows()