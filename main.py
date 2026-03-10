import cv2
from ultralytics import YOLO
from dronekit import connect

# ==========================
# CONNECT KE AUTOPILOT
# ==========================

print("Connecting to vehicle...")

vehicle = connect('127.0.0.1:14550', wait_ready=True)

print("Connected!")

# ==========================
# LOAD MODEL YOLO
# ==========================

model = YOLO("best.pt")

# ==========================
# CAMERA
# ==========================

cap = cv2.VideoCapture("http://192.168.1.5:8080/video")

FRAME_CENTER_TOLERANCE = 40

# ==========================
# CONTROL FUNCTION
# ==========================

def steer_left():
    vehicle.channels.overrides['1'] = 1400

def steer_right():
    vehicle.channels.overrides['1'] = 1600

def steer_straight():
    vehicle.channels.overrides['1'] = 1500

def throttle_forward():
    vehicle.channels.overrides['3'] = 1600

# ==========================
# MAIN LOOP
# ==========================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame)

    annotated = results[0].plot()

    red_x = None
    green_x = None
    black_area = 0

    boxes = results[0].boxes

    for box in boxes:

        cls = int(box.cls)
        name = model.names[cls]

        x1,y1,x2,y2 = box.xyxy[0]

        x_center = (x1 + x2) / 2
        area = (x2-x1)*(y2-y1)

        if name == "red_ball":
            red_x = x_center

        elif name == "green_ball":
            green_x = x_center

        elif name == "black_ball":
            black_area = area

    frame_center = frame.shape[1] / 2

    # ======================
    # DECISION MAKING
    # ======================

    if red_x and green_x:

        path_center = (red_x + green_x) / 2

        if path_center < frame_center - FRAME_CENTER_TOLERANCE:

            print("Go Left")
            steer_left()

        elif path_center > frame_center + FRAME_CENTER_TOLERANCE:

            print("Go Right")
            steer_right()

        else:

            print("Go Straight")
            steer_straight()

    elif red_x:

        print("Only Red → Turn Right")
        steer_right()

    elif green_x:

        print("Only Green → Turn Left")
        steer_left()

    elif black_area > 15000:

        print("Obstacle detected")

        steer_left()

    throttle_forward()

    cv2.imshow("USV Navigation", annotated)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
vehicle.close()