import cv2
from ultralytics import YOLO
import numpy as np
import time
from dronekit import connect, VehicleMode, Command
from pymavlink import mavutil

# ==============================
# 1. KONEKSI KE MISSION PLANNER (TCP)
# ==============================
print("Connecting to vehicle...")
# Menggunakan TCP 5760 sesuai koneksi Mission Planner kamu yang berhasil
vehicle = connect('tcp:127.0.0.1:5762', wait_ready=True)
print("Connected to Ship! Status:", vehicle.system_status.state)

# ==============================
# LOAD YOLO MODEL
# ==============================
model = YOLO("best.pt")

# ==============================
# CAMERA SETUP
# ==============================
cap = cv2.VideoCapture(1)

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

# Channel 1: Steering, Channel 3: Throttle
PWM_LEFT   = 1400
PWM_CENTER = 1500
PWM_RIGHT  = 1600
PWM_THROTTLE_GO   = 1600 # Kecepatan maju
PWM_THROTTLE_STOP = 1500 # Netral/Berhenti

NO_DETECTION_TIMEOUT       = 50.0 # Aku tambah jadi 10 detik biar gak gampang shutdown
NO_DETECTION_ORBIT_TIMEOUT = 15.0
ORBIT_CONFIRM_NEEDED       = 5

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
# AUTOPILOT CONTROL (Real DroneKit)
# ==============================
def set_steering(pwm):
    # Channel 1 biasanya untuk Steering/Servo di Rover/Ship
    vehicle.channels.overrides['1'] = pwm
    print(f"Setir ke: {pwm}")

def throttle_forward():
    # Channel 3 biasanya untuk Throttle/Gas
    vehicle.channels.overrides['3'] = PWM_THROTTLE_GO

def throttle_stop():
    vehicle.channels.overrides['3'] = PWM_THROTTLE_STOP
    # Bersihkan semua override saat berhenti
    vehicle.channels.overrides = {}

# ==============================
# DISTANCE ESTIMATION
# ==============================
def estimate_distance(box):
    x1, y1, x2, y2 = box
    diameter = max(x2 - x1, y2 - y1)
    if diameter == 0:
        return None
    return (REAL_BALL_DIAMETER * FOCAL_LENGTH) / diameter

# GERAK
def arm_and_takeoff(altitude):
    while not vehicle.is_armable:
        print("waiting to be armable")
        # time.sleep(1)
    print("Arming motors")
    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True

    print("Taking Off")
    vehicle.simple_takeoff(altitude)

    while True:
        v_alt = vehicle.location.global_relative_frame.alt
        print(">> Altitude = %.1f m" % v_alt)
        if v_alt >= altitude - 1.0:
            print("Target altitude reached")
            break


def add_last_waypoint_to_mission(  wp_Last_Latitude,
        wp_Last_Longitude,
        wp_Last_Altitude):  # --- [m]    Target Altitude
    """
    Upload the mission with the last WP as given and outputs the ID to be set
    """
    # Get the set of commands from the vehicle
    cmds = vehicle.commands
    cmds.download()
    cmds.wait_ready()

    # Save the vehicle commands to a list
    missionlist = []
    for cmd in cmds:
        missionlist.append(cmd)

    # Modify the mission as needed. For example, here we change the
    wpLastObject = Command(0, 0, 0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT, mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                           0, 0, 0, 0, 0, 0,
                           wp_Last_Latitude, wp_Last_Longitude, wp_Last_Altitude)
    missionlist.append(wpLastObject)

    # Clear the current mission (command is sent when we call upload())
    cmds.clear()

    # Write the modified mission and flush to the vehicle
    for cmd in missionlist:
        cmds.add(cmd)
    cmds.upload()

    return (cmds.count)


def gerak(vx, vy):
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,
        0, 0,
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,
        0b0000111111000111,
        0, 0, 0,
        vx, vy, 0,
        0, 0, 0,
        0, 0)
    vehicle.send_mavlink(msg)
    vehicle.flush()

# ==============================
# MAIN LOOP
# ==============================

perintah_sebelumnya = "lurus"
i_hilang = 0
batas_hilang = 500 # Sekitar 1-2 detik kalau FPS kamu 20-30

try:
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

        red_x = None; red_distance = 9999
        green_x = None; green_distance = 9999
        black_x = None; black_distance = 9999

        # PROCESS DETECTIONS
        for box in boxes:
            cls  = int(box.cls)
            name = model.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x_center = (x1 + x2) / 2
            distance = estimate_distance((x1, y1, x2, y2))
            
# ==============================
        # LOGIKA NAVIGATION (Berdasarkan Pseudocode)
        # ==============================
        steering = PWM_CENTER
        
        # 1. CEK JIKA ADA GERBANG (MERAH & HIJAU)
        if red_x is not None and green_x is not None:
            i_hilang = 0 # Reset counter karena target ketemu
            center_gate = (red_x + green_x) / 2
            
            if center_gate < frame_center - FRAME_CENTER_TOLERANCE:
                steering = PWM_LEFT
                perintah_sebelumnya = "kiri"
            elif center_gate > frame_center + FRAME_CENTER_TOLERANCE:
                steering = PWM_RIGHT
                perintah_sebelumnya = "kanan"
            else:
                steering = PWM_CENTER
                perintah_sebelumnya = "lurus"

        # 2. HANYA ADA BOLA MERAH
        elif red_x is not None:
            i_hilang = 0
            steering = PWM_RIGHT
            perintah_sebelumnya = "kanan"

        # 3. HANYA ADA BOLA HIJAU
        elif green_x is not None:
            i_hilang = 0
            steering = PWM_LEFT
            perintah_sebelumnya = "kiri"

        # 4. HANYA ADA BOLA HITAM (OBSTACLE)
        elif black_x is not None:
            i_hilang = 0
            # Sesuai pseudocode: jika di kiri frame, belok kiri makin jauh
            if black_x < 0.75 * frame_center:
                steering = PWM_LEFT
                perintah_sebelumnya = "kiri"
            else:
                steering = PWM_RIGHT
                perintah_sebelumnya = "kanan"

        # 5. BOLA HILANG (Gunakan Perintah Sebelumnya)
        else:
            i_hilang += 1
            print(f"Target Hilang! i={i_hilang}")
            
            if i_hilang >= batas_hilang:
                nav_state = STATE_SHUTDOWN # Panggil fungsi berhenti
                print(">>> EMERGENCY: Target Hilang Terlalu Lama!")
            else:
                # Jalankan perintah terakhir
                if perintah_sebelumnya == "kiri": steering = PWM_LEFT
                elif perintah_sebelumnya == "kanan": steering = PWM_RIGHT
                else: steering = PWM_CENTER

        # ==============================
        # EKSEKUSI GERAK KE KAPAL
        # ==============================
        if nav_state == STATE_SHUTDOWN:
            throttle_stop()
            set_steering(PWM_CENTER)
            break
        else:
            set_steering(steering)
            # Kamu bisa pakai gerak(vx, vy) di sini
            # vx = 1.5 m/s maju, vy = 0 (karena belok pakai setir)
            gerak(1.5, 0)

        # UPDATE DETECTION TIMER
        any_detected = (red_x is not None or green_x is not None or black_x is not None)
        if any_detected:
            last_detection_time = time.time()

        time_since_detection = time.time() - last_detection_time

        # STATE LOGIC & NAVIGATION
        gate_visible  = (red_x is not None and green_x is not None)
        obstacle_near = (black_x is not None and black_distance < OBSTACLE_DISTANCE_THRESHOLD)
        black_ratio   = (black_x / frame_width) if black_x is not None else None
        
        timeout_limit = NO_DETECTION_ORBIT_TIMEOUT if nav_state == STATE_CIRC_ORBIT else NO_DETECTION_TIMEOUT

        # Decision Making
        steering = PWM_CENTER

        if time_since_detection >= timeout_limit:
            nav_state = STATE_SHUTDOWN
        elif nav_state == STATE_GATE:
            if obstacle_near and not gate_visible:
                nav_state = STATE_CIRC_START
            elif gate_visible:
                path_center = (red_x + green_x) / 2
                steering = PWM_LEFT if path_center < frame_center - FRAME_CENTER_TOLERANCE else (PWM_RIGHT if path_center > frame_center + FRAME_CENTER_TOLERANCE else PWM_CENTER)
            elif red_x: steering = PWM_RIGHT
            elif green_x: steering = PWM_LEFT
        
        elif nav_state == STATE_CIRC_START:
            steering = PWM_LEFT
            if black_ratio and black_ratio >= ORBIT_RIGHT_ZONE:
                orbit_confirm_count += 1
                if orbit_confirm_count >= ORBIT_CONFIRM_NEEDED:
                    nav_state = STATE_CIRC_ORBIT
            if gate_visible: nav_state = STATE_GATE

        elif nav_state == STATE_CIRC_ORBIT:
            if gate_visible or black_x is None:
                nav_state = STATE_GATE
            else:
                steering = PWM_LEFT if black_ratio < ORBIT_KEEP_MIN else (PWM_CENTER if black_ratio > ORBIT_KEEP_MAX else PWM_RIGHT)

        # OUTPUT KE KAPAL
        if nav_state == STATE_SHUTDOWN:
            throttle_stop()
            set_steering(PWM_CENTER)
            print("!!! EMERGENCY SHUTDOWN !!!")
            break # Keluar loop
        else:
            set_steering(steering)
            throttle_forward()

        # Tampilkan hasil di layar
        cv2.imshow("Naval Cam AI", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

except Exception as e:
    print(f"Error: {e}")

finally:
    # Selalu hentikan kapal sebelum script mati
    print("Cleaning up...")
    throttle_stop()
    cap.release()
    cv2.destroyAllWindows()
    vehicle.close()