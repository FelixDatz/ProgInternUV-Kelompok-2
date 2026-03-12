import cv2
import numpy as np
import time
import collections
from dronekit import connect, VehicleMode
from ultralytics import YOLO

# --- 1. FIX COMPATIBILITY (DARI KODE SENIOR) ---
try:
    from collections import abc
    collections.MutableMapping = abc.MutableMapping
except:
    pass

# --- 2. SETUP KONEKSI & SISTEM ---
print("Menghubungkan ke kapal...")
# Menggunakan koneksi TCP untuk simulasi
vehicle = connect('tcp:127.0.0.1:5762', wait_ready=True)

def prepare_vehicle():
    """Fungsi Arming otomatis agar tidak pusing klik di Mission Planner"""
    print("Menunggu kapal siap di-arm...")
    while not vehicle.is_armable:
        time.sleep(1)
    
    # Set ke mode MANUAL untuk kontrol via channels.overrides
    vehicle.mode = VehicleMode("MANUAL")
    vehicle.armed = True
    
    while not vehicle.armed:
        print("Mencoba Arming...")
        time.sleep(1)
    print("KAPAL ARMED & READY!")

# Jalankan persiapan kapal
prepare_vehicle()

# Load Model YOLO & Kamera
model = YOLO("best.pt")
cap = cv2.VideoCapture(0)

# --- 3. FUNGSI KONTROL (STRUKTUR TEMAN - RAPI) ---
def set_control(steer, throttle=1700):
    """
    1500 = Netral
    Steer: < 1500 Kiri, > 1500 Kanan
    Throttle: > 1500 Maju
    """
    vehicle.channels.overrides = {'1': steer, '3': throttle}

# --- 4. VARIABEL LOGIKA (MEMORI LINTASAN) ---
perintah_terakhir = "lurus"
i_hilang = 0
FRAME_TOLERANCE = 40 # Toleransi tengah frame

print("Sistem Navigasi Aktif! Tekan 'q' untuk berhenti.")

try:
    while True:
        # PERSEPSI (Kamera)
        ret, frame = cap.read()
        if not ret: break

        frame_width = frame.shape[1]
        frame_center = frame_width / 2
        
        # Deteksi YOLO
        results = model(frame, stream=True)
        red_x, green_x, black_x = None, None, None
        
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                name = model.names[cls]
                # Hitung titik tengah objek
                x_mid = float((box.xyxy[0][0] + box.xyxy[0][2]) / 2)
                
                if name == "bola_merah": red_x = x_mid
                elif name == "bola_hijau": green_x = x_mid
                elif name == "bola_hitam": black_x = x_mid

        # --- DECISION MAKING (LOGIKA TERBAIK) ---
        steer = 1500
        throttle = 1700 # Kecepatan jelajah

        # KONDISI 1: ADA BOLA HITAM (Prioritas Utama - Memutari Kanan)
        if black_x is not None:
            i_hilang = 0
            steer = 1700 # Belok tajam kanan untuk memutari
            perintah_terakhir = "kanan"
            status = "MEMUTARI OBSTACLE"

        # KONDISI 2: ADA GAWANG LENGKAP (Merah & Hijau)
        elif red_x is not None and green_x is not None:
            i_hilang = 0
            path_center = (red_x + green_x) / 2
            
            if path_center < (frame_center - FRAME_TOLERANCE):
                steer = 1400
                perintah_terakhir = "kiri"
            elif path_center > (frame_center + FRAME_TOLERANCE):
                steer = 1600
                perintah_terakhir = "kanan"
            else:
                steer = 1500
                perintah_terakhir = "lurus"
            status = "NAVIGASI GAWANG"

        # KONDISI 3: HANYA HIJAU (Sesuai Gambar: Harus Belok Kiri agar Hijau tetap di Kanan)
        elif green_x is not None:
            i_hilang = 0
            # Jika hijau terlalu masuk ke tengah area kiri kapal
            if green_x < (frame_width * 0.7):
                steer = 1400 
                perintah_terakhir = "kiri"
            else:
                steer = 1500
            status = "KOREKSI HIJAU (TIANG KANAN)"

        # KONDISI 4: HANYA MERAH (Harus Belok Kanan agar Merah tetap di Kiri)
        elif red_x is not None:
            i_hilang = 0
            # Jika merah terlalu masuk ke tengah area kanan kapal
            if red_x > (frame_width * 0.3):
                steer = 1600
                perintah_terakhir = "kanan"
            else:
                steer = 1500
            status = "KOREKSI MERAH (TIANG KIRI)"

        # KONDISI 5: TIDAK ADA TARGET (Insting Mencari)
        else:
            i_hilang += 1
            throttle = 1650 # Kurangi kecepatan saat mencari
            
            if i_hilang < 40: # Memori manuver (biar belokan tidak patah)
                if perintah_terakhir == "kiri": steer = 1400
                elif perintah_terakhir == "kanan": steer = 1600
                status = "TARGET HILANG (FOLLOWING LAST)"
            else:
                steer = 1500
                status = "MENCARI TARGET (LURUS)"

        # --- EKSEKUSI ---
        set_control(steer, throttle)
        
        # Monitor di Terminal
        print(f"Status: {status} | Steer: {steer} | i_lost: {i_hilang}")

        # Visualisasi (Pakai plot() agar kotak deteksi muncul)
        # Re-fetching results for plotting convenience
        res_plot = model(frame)
        cv2.imshow("Mata Kapal Otonom KKCTBN", res_plot[0].plot())
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"Error: {e}")

finally:
    # CLEANUP
    print("\nMematikan Sistem...")
    set_control(1500, 1500) # Stop Kapal
    vehicle.close()
    cap.release()
    cv2.destroyAllWindows()