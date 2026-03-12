import cv2
import numpy as np
import time
from dronekit import connect, VehicleMode
from ultralytics import YOLO

# 1. SETUP KONEKSI (DroneKit)
print("Menghubungkan ke kapal...")
try:
    vehicle = connect('tcp:127.0.0.1:5762', wait_ready=True)
    print(f"Koneksi Berhasil!\nStatus Kapal: {vehicle.system_status.state}")
except Exception as e:
    print(f"Gagal konek: {e}")
    exit() # Hentikan program jika koneksi gagal

# 2. SETUP MODEL & KAMERA
print("Memuat model YOLO dan Kamera...")
model = YOLO("best.pt")
cap = cv2.VideoCapture(0)

# 3. SETUP VARIABEL LOGIKA TIM (Pseudocode)
perintah_sebelumnya = "lurus"
i_hilang = 0
batas_hilang = 130
print("Sistem Siap! Tekan 'q' pada jendela kamera untuk keluar.")

try:
    while True:
        # --- PERSEPSI ---
        ret, frame = cap.read()
        if not ret:
            print("Kamera error!")
            break

        frame_height, frame_width = frame.shape[:2]
        frame_center = 0.5 * frame_width 

        results = model(frame, stream=True)
        
        red_x = None
        green_x = None
        black_x = None

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                name = model.names[cls]
                x1, y1, x2, y2 = box.xyxy[0]
                x_mid = (x1 + x2) / 2

                if name == "bola_merah":
                    red_x = float(x_mid)
                elif name == "bola_hijau":
                    green_x = float(x_mid)
                elif name == "bola_hitam":
                    black_x = float(x_mid)

        # --- LOGIKA KEPUTUSAN (PINDAH KE DALAM WHILE) ---
        steering = 1500 
        throttle = 1700 

        if red_x is not None and green_x is not None:
            i_hilang = 0 
            center_gate = (red_x + green_x) / 2
            if center_gate < frame_center:
                steering = 1400 
                perintah_sebelumnya = "kiri"
            elif center_gate > frame_center:
                steering = 1600 
                perintah_sebelumnya = "kanan"
            else:
                steering = 1500
                perintah_sebelumnya = "lurus"

        elif red_x is not None:
            i_hilang = 0
            steering = 1600 
            perintah_sebelumnya = "kanan"

        elif green_x is not None:
            i_hilang = 0
            steering = 1400 
            perintah_sebelumnya = "kiri"

        elif black_x is not None:
            i_hilang = 0
            if black_x < 0.75 * frame_center:
                steering = 1400 
                perintah_sebelumnya = "kiri"
            else:
                steering = 1600 
                perintah_sebelumnya = "kanan"

        #else: # No Ball
         #   i_hilang += 1
          #  if i_hilang >= batas_hilang:
           #     print(">>> EMERGENCY: Target Hilang! Berhenti.")
            #    throttle = 1500 
             #   steering = 1500
            #else:
             #   if perintah_sebelumnya == "kiri": steering = 1400
              #  elif perintah_sebelumnya == "kanan": steering = 1600
               # else: steering = 1500
            
        else: # No Ball (Kondisi saat tidak ada bola terdeteksi)
            i_hilang += 1
            
            if i_hilang > 50:
                # Jika sudah terlalu lama hilang (50 frame), luruskan kapal
                steering = 1500
                perintah_sebelumnya = "mencari (lurus)"
            else:
                # Jika baru sebentar, lanjutkan belokan terakhir
                if perintah_sebelumnya == "kiri": 
                    steering = 1400
                elif perintah_sebelumnya == "kanan": 
                    steering = 1600
                else: 
                    steering = 1500
            
            print(f">>> Target Hilang {i_hilang} frame, Status: {perintah_sebelumnya}")

        # --- EKSEKUSI (PINDAH KE DALAM WHILE) ---
        vehicle.channels.overrides['1'] = steering 
        vehicle.channels.overrides['3'] = throttle 
        
        print(f"Status: {perintah_sebelumnya} | Steering: {steering} | i: {i_hilang}")

        # Visualisasi
        cv2.imshow("Mata Kapal Otonom", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"Terjadi error dalam loop: {e}")

finally:
    # --- CLEANUP (Tetap di luar loop) ---
    print("\nMenutup sistem dan menghentikan kapal...")
    try:
        vehicle.channels.overrides = {'1': 1500, '3': 1500}
        vehicle.close()
    except:
        pass
    cap.release()
    cv2.destroyAllWindows()