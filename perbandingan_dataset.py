import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

# --- KONFIGURASI PERTARUNGAN ---
# Kita cuma ambil 3 huruf biar super cepat
KELAS_ADUAN = ['A', 'B', 'C'] 
EPOCH_ADUAN = 5
IMG_SIZE = 128

# Path Folder (SESUAIKAN NAMA FOLDERMU!)
DIR_ASLI = 'asl_alphabet_train'      # Dataset Kaggle (Warna)
DIR_BARU = 'dataset_skeleton_kag'    # Dataset Kita (Hitam Putih)

# Cek dulu
if not os.path.exists(DIR_ASLI) or not os.path.exists(DIR_BARU):
    print("❌ Error: Salah satu folder dataset gak ketemu. Cek nama folder!")
    exit()

# Fungsi Pembuat Model (Biar Adil, Arsitekturnya Sama Persis)
def bikin_model():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(len(KELAS_ADUAN), activation='softmax') # Output cuma 3 kelas
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ==========================================
# RONDE 1: DATASET LAMA (RAW IMAGE)
# ==========================================
print("\n🥊 RONDE 1: Training pake Gambar Asli (Warna)...")
datagen_raw = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_raw = datagen_raw.flow_from_directory(
    DIR_ASLI, target_size=(IMG_SIZE, IMG_SIZE), classes=KELAS_ADUAN, 
    batch_size=32, subset='training', shuffle=True
)
val_raw = datagen_raw.flow_from_directory(
    DIR_ASLI, target_size=(IMG_SIZE, IMG_SIZE), classes=KELAS_ADUAN, 
    batch_size=32, subset='validation'
)

model_raw = bikin_model()
hist_raw = model_raw.fit(train_raw, validation_data=val_raw, epochs=EPOCH_ADUAN, verbose=1)

# ==========================================
# RONDE 2: DATASET KITA (SKELETON)
# ==========================================
print("\n🥊 RONDE 2: Training pake Dataset Skeleton...")
datagen_skel = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_skel = datagen_skel.flow_from_directory(
    DIR_BARU, target_size=(IMG_SIZE, IMG_SIZE), classes=KELAS_ADUAN, 
    batch_size=32, subset='training', shuffle=True
)
val_skel = datagen_skel.flow_from_directory(
    DIR_BARU, target_size=(IMG_SIZE, IMG_SIZE), classes=KELAS_ADUAN, 
    batch_size=32, subset='validation'
)

model_skel = bikin_model()
hist_skel = model_skel.fit(train_skel, validation_data=val_skel, epochs=EPOCH_ADUAN, verbose=1)

# ==========================================
# HASIL AKHIR: GAMBAR GRAFIK
# ==========================================
print("\n📊 Menggambar Grafik Perbandingan...")
acc_raw = hist_raw.history['val_accuracy']
acc_skel = hist_skel.history['val_accuracy']
epochs = range(1, EPOCH_ADUAN + 1)

plt.figure(figsize=(10, 6))

# Garis Merah (Lama)
plt.plot(epochs, acc_raw, 'r--o', label='Metode Lama (Raw Image)')
# Garis Biru (Baru)
plt.plot(epochs, acc_skel, 'b-o', linewidth=3, label='Metode Usulan (Skeleton Hybrid)')

plt.title(f'Pertarungan Efisiensi: Raw vs Skeleton (Kelas {KELAS_ADUAN})')
plt.xlabel('Epoch')
plt.ylabel('Akurasi Validasi')
plt.legend()
plt.grid(True, alpha=0.3)

plt.savefig('hasil_pertarungan_real.png')
print("✅ Selesai! Cek file 'hasil_pertarungan_real.png'.")
plt.show()
