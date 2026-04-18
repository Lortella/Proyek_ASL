import cv2
import os
import numpy as np
import mediapipe as mp

# --- KONFIGURASI ---
INPUT_DIR = "asl_alphabet_train"  
# Folder baru hasil konversi (Otomatis dibuat)
OUTPUT_DIR = "dataset_skeleton_kag" 
IMG_SIZE = 128

# Setup MediaPipe (Mode Static untuk Foto)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print(f"🏭 Memulai pabrik konversi dari '{INPUT_DIR}' ke '{OUTPUT_DIR}'...")

total_converted = 0
skipped = 0

# Loop semua folder (A, B, C...)
for label in os.listdir(INPUT_DIR):
    input_path = os.path.join(INPUT_DIR, label)
    if not os.path.isdir(input_path): continue
    
    # Buat folder output yang sesuai
    output_path = os.path.join(OUTPUT_DIR, label)
    if not os.path.exists(output_path): os.makedirs(output_path)
    
    print(f"📂 Memproses huruf: {label}...")
    
    # Loop semua file dalam folder huruf
    files = os.listdir(input_path)
    # Opsional: Kalau kebanyakan, batasi misal [:500] biar cepet
    for img_name in files: 
        img_full_path = os.path.join(input_path, img_name)
        
        # Baca Gambar
        img = cv2.imread(img_full_path)
        if img is None: continue
        
        # Deteksi Tangan
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        
        if results.multi_hand_landmarks:
            for lm in results.multi_hand_landmarks:
                # Buat Kanvas Hitam
                h, w, _ = img.shape
                skeleton_img = np.zeros((h, w, 3), dtype=np.uint8)
                
                # Gambar Garis Putih Tebal
                mp_drawing.draw_landmarks(
                    skeleton_img, lm, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255,255,255), thickness=5, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255,255,255), thickness=5, circle_radius=2)
                )
                
                # Crop agar pas di tangan (Biar data rapi)
                x_list = [l.x for l in lm.landmark]
                y_list = [l.y for l in lm.landmark]
                x_min, x_max = int(min(x_list)*w), int(max(x_list)*w)
                y_min, y_max = int(min(y_list)*h), int(max(y_list)*h)
                
                # Padding aman
                pad = 20
                x_min = max(0, x_min-pad); x_max = min(w, x_max+pad)
                y_min = max(0, y_min-pad); y_max = min(h, y_max+pad)
                
                crop = skeleton_img[y_min:y_max, x_min:x_max]
                
                if crop.size != 0:
                    # Resize final ke 128x128
                    final = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
                    
                    # Simpan
                    save_name = os.path.join(output_path, img_name)
                    cv2.imwrite(save_name, final)
                    total_converted += 1
        else:
            skipped += 1

print("✅ SELESAI!")
print(f"Total Sukses: {total_converted} gambar")
print(f"Gagal/Skip  : {skipped} gambar (Wajar karena foto buram/tangan kepotong)")
