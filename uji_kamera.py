import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import time

# --- KONFIGURASI SISTEM ---
MODEL_PATH = 'asl_cnn_skeleton.h5'
IMG_SIZE = 128
CONFIDENCE_THRESHOLD = 0.7 # Ambang batas deteksi tangan

# 1. INISIALISASI MODEL & LABEL
print("Info: Memuat model CNN dan label kelas...")
try:
    with open("labels.txt", "r") as f:
        LABELS = f.read().splitlines()
    model = load_model(MODEL_PATH)
    print("Sukses: Model berhasil dimuat. Memulai stream kamera...")
except Exception as e:
    print(f"Error: Gagal memuat model atau label. {e}")
    exit()

# 2. INISIALISASI MEDIAPIPE
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=CONFIDENCE_THRESHOLD)

# Setup Kamera
cap = cv2.VideoCapture(0)
prev_frame_time = 0
new_frame_time = 0

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # Mirroring frame (membalik horizontal) agar natural bagi pengguna
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    # Perhitungan FPS (Frames Per Second)
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    
    # Konversi BGR ke RGB untuk MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    
    # Inisialisasi kanvas kosong (Blank Canvas) untuk input model
    ai_input = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    prediction_text = "Menunggu Input..."
    conf_score = 0
    
    if results.multi_hand_landmarks:
        for lm in results.multi_hand_landmarks:
            # A. VISUALISASI PADA FRAME PENGGUNA
            mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
            
            # B. EKSTRAKSI REGION OF INTEREST (ROI)
            # Mendapatkan koordinat bounding box tangan
            x_list = [l.x for l in lm.landmark]
            y_list = [l.y for l in lm.landmark]
            x1, x2 = int(min(x_list)*w), int(max(x_list)*w)
            y1, y2 = int(min(y_list)*h), int(max(y_list)*h)
            
            # Menambahkan padding agar jari tidak terpotong
            pad = 30
            x1 = max(0, x1-pad); x2 = min(w, x2+pad)
            y1 = max(0, y1-pad); y2 = min(h, y2+pad)
            
            # C. REKONSTRUKSI SKELETON (PREPROCESSING)
            # Menggambar ulang landmarks pada kanvas hitam bersih
            temp_black = np.zeros((h, w, 3), dtype=np.uint8)
            mp_drawing.draw_landmarks(
                temp_black, lm, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255,255,255), thickness=4, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255,255,255), thickness=4, circle_radius=2)
            )
            
            # Crop area tangan dari kanvas hitam
            skeleton_crop = temp_black[y1:y2, x1:x2]
            
            # Validasi ukuran crop (mencegah error jika tangan keluar frame)
            if skeleton_crop.size != 0:
                # Resize ke dimensi input model (128x128)
                ai_input = cv2.resize(skeleton_crop, (IMG_SIZE, IMG_SIZE))
                
                # Normalisasi piksel (0-1) & Penambahan dimensi batch
                img_pred = ai_input.astype('float32') / 255.0
                img_pred = np.expand_dims(img_pred, axis=0)
                
                # D. PROSES INFERENCE (PREDIKSI)
                pred = model.predict(img_pred, verbose=0)[0]
                idx = np.argmax(pred)
                prediction_text = LABELS[idx]
                conf_score = pred[idx] * 100
                
                # Tampilkan hasil prediksi di layar
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{prediction_text} {conf_score:.1f}%", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Tampilkan FPS
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Tampilkan Jendela Aplikasi
    # Window 1: Tampilan Interaktif untuk Pengguna
    cv2.imshow('Sistem Pengenalan ASL - Tampilan Pengguna', frame)
    
    # Window 2: Visualisasi Input Model (Untuk keperluan debug/analisis)
    cv2.imshow('Input Preprocessing (Skeleton Extraction)', ai_input) 

    # Tombol 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
print("Info: Program dihentikan oleh pengguna.")
