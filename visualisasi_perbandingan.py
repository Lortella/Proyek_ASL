import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import os
import random

# --- KONFIGURASI ---
# Pastikan ini nama folder dataset ASLI kamu (yang berwarna)
DIR_ASLI = "asl_alphabet_train" 
IMG_SIZE = 128

# Cek folder
if not os.path.exists(DIR_ASLI):
    print(f"❌ Gak nemu folder '{DIR_ASLI}'. Sesuaikan namanya dulu!")
    exit()

# Setup MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# Ambil 1 Huruf Acak
huruf_acak = random.choice(os.listdir(DIR_ASLI))
folder_huruf = os.path.join(DIR_ASLI, huruf_acak)
file_acak = random.choice(os.listdir(folder_huruf))
path_img = os.path.join(folder_huruf, file_acak)

# 1. BACA GAMBAR ASLI
img_asli = cv2.imread(path_img)
img_rgb = cv2.cvtColor(img_asli, cv2.COLOR_BGR2RGB)

# 2. PROSES JADI SKELETON
results = hands.process(img_rgb)
h, w, _ = img_asli.shape
img_skeleton = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8) # Kanvas Hitam

if results.multi_hand_landmarks:
    for lm in results.multi_hand_landmarks:
        # Gambar di kanvas sementara seukuran asli
        temp_canvas = np.zeros((h, w, 3), dtype=np.uint8)
        mp_drawing.draw_landmarks(
            temp_canvas, lm, mp_hands.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(255,255,255), thickness=5, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255,255,255), thickness=5, circle_radius=2)
        )
        
        # Crop (Kita simulasi logika yang sama dgn training)
        x_list = [l.x for l in lm.landmark]
        y_list = [l.y for l in lm.landmark]
        x1, x2 = int(min(x_list)*w), int(max(x_list)*w)
        y1, y2 = int(min(y_list)*h), int(max(y_list)*h)
        
        pad = 20
        x1 = max(0, x1-pad); x2 = min(w, x2+pad)
        y1 = max(0, y1-pad); y2 = min(h, y2+pad)
        
        crop = temp_canvas[y1:y2, x1:x2]
        if crop.size != 0:
            img_skeleton = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))

# 3. PLOT PERBANDINGAN (TAMPILAN LAPORAN)
plt.figure(figsize=(10, 5))

# Kiri: Asli
plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title(f"Dataset Asli (Raw Image)\nHuruf: {huruf_acak}", fontsize=12)
plt.axis('off')

# Kanan: Skeleton
plt.subplot(1, 2, 2)
plt.imshow(img_skeleton)
plt.title(f"Dataset Baru (Skeleton Input)\nSiap masuk ke CNN", fontsize=12)
plt.axis('off')

# Simpan
nama_file = "perbandingan_dataset.png"
plt.savefig(nama_file, bbox_inches='tight')
print(f"✅ Gambar perbandingan disimpan: {nama_file}")
plt.show()
