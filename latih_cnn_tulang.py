import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

# --- KONFIGURASI MIXED PRECISION ---
# Mengaktifkan presisi 16-bit untuk mempercepat komputasi GPU dan efisiensi memori
try:
    mixed_precision.set_global_policy('mixed_float16')
    print("Info: Mixed Precision (16-bit) berhasil diaktifkan.")
except:
    print("Peringatan: Gagal mengaktifkan Mixed Precision, menggunakan mode default (32-bit).")

# --- KONFIGURASI PATH & PARAMETER ---
DATA_DIR = 'dataset_skeleton_kag' 
MODEL_NAME = 'asl_cnn_skeleton.h5'
IMG_SIZE = 128

# Batch Size diset tinggi (256) untuk memaksimalkan throughput GPU
# Jika terjadi error Out of Memory (OOM), turunkan menjadi 128 atau 64
BATCH_SIZE = 256 

if not os.path.exists(DATA_DIR):
    print(f"Error: Direktori {DATA_DIR} tidak ditemukan. Pastikan preprocessing sudah dijalankan.")
    exit()

# 1. DATA GENERATOR & AUGMENTASI
# Normalisasi nilai piksel ke rentang [0,1] dan augmentasi ringan
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=5,       # Rotasi ringan untuk variasi
    width_shift_range=0.05, # Pergeseran horizontal
    validation_split=0.1    # Pembagian data validasi 10%
)

print(f"Info: Memulai persiapan data dengan Batch Size: {BATCH_SIZE}")

train_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Menyimpan label kelas ke file teks untuk referensi saat pengujian
CLASSES = sorted(train_gen.class_indices.keys())
with open("labels.txt", "w") as f:
    f.write("\n".join(CLASSES))

# 2. ARSITEKTUR MODEL CNN
# Menggunakan arsitektur Sequential dengan 3 blok konvolusi
model = Sequential([
    # Blok 1: Ekstraksi Fitur Dasar
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2,2),
    
    # Blok 2: Ekstraksi Fitur Menengah
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    # Blok 3: Ekstraksi Fitur Kompleks
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    # Flatten & Fully Connected Layer
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5), # Dropout 50% untuk mencegah Overfitting
    
    # Output Layer (Softmax untuk klasifikasi multi-kelas)
    # dtype='float32' diperlukan untuk stabilitas numerik pada output layer
    Dense(len(CLASSES), activation='softmax', dtype='float32') 
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 3. PROSES PELATIHAN (TRAINING)
print("Memulai proses pelatihan model (Target: 8 Epoch)...")

history = model.fit(
    train_gen, 
    validation_data=val_gen, 
    epochs=8,            # Jumlah epoch disesuaikan dengan kecepatan konvergensi data skeleton
    verbose=1,
    workers=12,          # Menggunakan multi-threading CPU untuk pemuatan data
    use_multiprocessing=False 
)

# 4. PENYIMPANAN MODEL
model.save(MODEL_NAME)
print(f"Sukses: Model telah disimpan sebagai '{MODEL_NAME}'")

# 5. VISUALISASI HASIL EVALUASI
print("Membuat grafik evaluasi kinerja...")

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(12, 6))

# Subplot 1: Grafik Akurasi
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Grafik Akurasi Model')
plt.xlabel('Epoch')
plt.ylabel('Akurasi')

# Subplot 2: Grafik Loss (Error)
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Grafik Loss (Tingkat Kesalahan)')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# Simpan grafik ke file gambar
plt.savefig('grafik_evaluasi.png') 
print("Sukses: Grafik evaluasi disimpan sebagai 'grafik_evaluasi.png'")
plt.show()
