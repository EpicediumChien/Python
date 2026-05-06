import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from scipy import signal
import os
import glob
import random
import joblib
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
# Hard Balancing and Regularization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import GaussianNoise

# --- GPU MEMORY MANAGEMENT ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print(f"GPU Setup error: {e}")

# ==========================================
# 1. Configuration (Synced with Analysis)
# ==========================================
DATA_DIR = os.path.join('TrainingData', 'Static Sit')
TARGET_FREQ = 20        # Training at 20Hz
WINDOW_SIZE = 200       # 10 Seconds
STEP_SIZE = 10          # High overlap for more data
AUGMENTATION_COUNT = 1 
FEATURES = ['accX', 'accY', 'accZ', 'gyroX', 'gyroY', 'gyroZ', 'roll', 'pitch', 'yaw']

# # BPM Logic from Analysis
# BPM_NORMAL_MIN = 12.0
# BPM_NORMAL_MAX = 20.0

# ==========================================
# 2. Ported Pre-processing (Direct Port from Analysis)
# ==========================================

def apply_agc_logic(sig, fs=20, window_sec=10):
    """Mirror Analysis: 10s window to stabilize signals"""
    window_size = int(fs * window_sec)
    rolling_std = pd.Series(sig).rolling(window=window_size, center=True).std()
    rolling_std = rolling_std.bfill().ffill().replace(0, 1e-6)
    return sig / rolling_std.values

def clean_signal_logic(data, fs=20):
    cleaned_data = np.zeros_like(data)
    nyq = 0.5 * fs
    
    # 1. 你的原始濾波器 [0.08Hz - 0.85Hz]
    b, a = signal.butter(2, [0.08/nyq, 0.85/nyq], btype='band')
    
    # 2. 你的原始平滑視窗 (1.1s)
    savgol_win = 27 # 25Hz * 1.1s ≈ 27
    
    for i in range(data.shape[1]):
        # A. 帶通濾波
        feat_filt = signal.filtfilt(b, a, data[:, i])
        
        # B. 你的原始 AGC (10秒視窗)
        feat_agc = apply_agc_logic(feat_filt, fs=fs, window_sec=10)
        
        # C. 你的原始 SavGol
        try:
            cleaned_data[:, i] = signal.savgol_filter(feat_agc, window_length=savgol_win, polyorder=2)
        except:
            cleaned_data[:, i] = feat_agc
            
    return cleaned_data


# ==========================================
# 3. Frequency Detection for Labeling
# ==========================================

def get_dominant_bpm(window_data, fs=20):
    """Detects BPM for labeling using FFT"""
    sig = window_data[:, -1] # Use Acc Magnitude
    sig = sig - np.mean(sig) 
    n = len(sig)
    freqs = np.fft.rfftfreq(n, d=1/fs)
    fft_mag = np.abs(np.fft.rfft(sig))
    
    # Range synced with Filter: 0.08Hz to 0.85Hz
    idx = np.where((freqs >= 0.08) & (freqs <= 0.85))[0]
    if len(idx) == 0: return 15.0
    
    return freqs[idx][np.argmax(fft_mag[idx])] * 60

def generate_abnormal_by_bpm(window_data, fs=20):
    orig_bpm = get_dominant_bpm(window_data, fs)
    orig_len = len(window_data)
    
    # 嚴格定義異常頻率
    target_bpm = random.uniform(25, 40) if random.random() > 0.5 else random.uniform(6, 10)
    
    ratio = target_bpm / orig_bpm
    new_len = max(int(orig_len / ratio), 1)
    resampled = signal.resample(window_data, new_len)
    
    # 這裡只做重複或截斷，不要做 **1.4 扭曲
    if ratio > 1.0:
        output = np.tile(resampled, (int(np.ceil(orig_len / new_len)), 1))[:orig_len, :]
    else:
        output = resampled[:orig_len, :]
        if len(output) < orig_len:
            output = np.pad(output, ((0, orig_len - len(output)), (0, 0)), mode='edge')

    # 只加入極小量的噪聲 (模擬感測器底噪)
    return output + np.random.normal(0, 0.005, output.shape)

# ==========================================
# 4. Global Scaling Preparation
# ==========================================

def process_single_file(file_path):
    try:
        df = pd.read_csv(file_path)
        df.columns = [c.lower() for c in df.columns]
        ts_col = 'unix_timestamp' if 'unix_timestamp' in df.columns else 'timestamp'
        df['dt'] = pd.to_datetime(df[ts_col], unit='ms')
        df = df.set_index('dt')
        df_res = df.resample('50ms').mean().interpolate().ffill().bfill() # 20Hz

        acc_mag = np.linalg.norm(df_res[[c for c in df_res.columns if 'acc' in c][:3]].values, axis=1, keepdims=True)
        gyro_mag = np.linalg.norm(df_res[[c for c in df_res.columns if 'gyro' in c][:3]].values, axis=1, keepdims=True)
        
        raw_data = df_res.reindex(columns=[f.lower() for f in FEATURES]).values.astype(np.float32)
        combined_data = np.hstack([raw_data, acc_mag, gyro_mag]) 
        processed_data = clean_signal_logic(combined_data, fs=TARGET_FREQ)

        x_list, y_list = [], []
        for i in range(0, len(processed_data) - WINDOW_SIZE, STEP_SIZE):
            window = processed_data[i : i + WINDOW_SIZE]
            current_bpm = get_dominant_bpm(window, fs=TARGET_FREQ)
            
            # Per-Window Normalization
            window_norm = (window - np.mean(window, axis=0)) / (np.std(window, axis=0) + 1e-6)

            if 12.0 <= current_bpm <= 20.0:
                x_list.append(window_norm); y_list.append(0)
                for _ in range(AUGMENTATION_COUNT):
                    ab_win = generate_abnormal_by_bpm(window, fs=TARGET_FREQ)
                    ab_norm = (ab_win - np.mean(ab_win, axis=0)) / (np.std(ab_win, axis=0) + 1e-6)
                    x_list.append(ab_norm); y_list.append(1)
            else:
                x_list.append(window_norm); y_list.append(1)
        return np.array(x_list), np.array(y_list)
    except Exception as e:
        return None, None

# ==========================================
# 4. Model & Training
# ==========================================
# 1. FIXED MODEL: Added L2, Noise, and Hybrid Pooling
def build_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    # Noise layer to prevent memorizing specific CSV spikes
    x = GaussianNoise(0.01)(inputs)

    # Conv Block 1
    x = layers.Conv1D(32, 7, activation='relu', padding='same', kernel_regularizer=l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.SpatialDropout1D(0.2)(x)
    x = layers.MaxPooling1D(2)(x)

    # Conv Block 2
    x = layers.Conv1D(64, 5, activation='relu', padding='same', kernel_regularizer=l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    # Hybrid Pooling (This helps CNNs understand 'Rhythm' better)
    avg_p = layers.GlobalAveragePooling1D()(x)
    max_p = layers.GlobalMaxPooling1D()(x)
    combined = layers.concatenate([avg_p, max_p])

    x = layers.Dense(32, activation='relu', kernel_regularizer=l2(0.01))(combined)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), 
                  loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05), 
                  metrics=['accuracy'])
    return model

# 2. FIXED DATA: Forced 1:1 Balancing
if __name__ == "__main__":
    file_list = glob.glob(os.path.join(DATA_DIR, '*.csv'))
    random.shuffle(file_list)
    
    def collect_and_balance(files):
        x_raw, y_raw = [], []
        for f in files:
            xf, yf = process_single_file(f)
            if xf is not None: x_raw.append(xf); y_raw.append(yf)
        
        X = np.concatenate(x_raw)
        y = np.concatenate(y_raw)
        
        # --- HARD BALANCING ---
        idx0 = np.where(y == 0)[0]
        idx1 = np.where(y == 1)[0]
        min_s = min(len(idx0), len(idx1))
        np.random.shuffle(idx0); np.random.shuffle(idx1)
        
        balanced_idx = np.concatenate([idx0[:min_s], idx1[:min_s]])
        np.random.shuffle(balanced_idx)
        return X[balanced_idx], y[balanced_idx]

    print("Processing and Balancing...")
    split = int(len(file_list) * 0.8)
    X_train, y_train = collect_and_balance(file_list[:split])
    X_test, y_test = collect_and_balance(file_list[split:])

    # --- REMOVED Global StandardScaler ---
    
    model = build_model((WINDOW_SIZE, X_train.shape[2]))
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=64,
        validation_data=(X_test, y_test),
        callbacks=[
            EarlyStopping(patience=15, restore_best_weights=True),
            ReduceLROnPlateau(patience=5, factor=0.5)
        ]
    )

    # ==========================================
    # 5. Evaluation & Results Visualization
    # ==========================================
    print("\n--- Generating Training Results ---")

    target_names = ['Normal (Eupnea)', 'Abnormal (Dyspnea/Apnea)']

    # Results Visualization (Standardized)
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.4).astype(int)
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Abnormal']))

    # 3. Plotting Training History
    plt.figure(figsize=(12, 5))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 4. Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()

    # 5. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()

    # ==========================================
    # 6. Save Model Artifacts
    # ==========================================
    os.makedirs('models', exist_ok=True)
    model.save('models/respiration_CNN_classifier.keras')
    # joblib.dump(scaler, 'models/CNN_scaler.pkl')
    print(f"\nTraining Complete. Model saved to 'models/respiration_CNN_classifier.keras' (AUC: {roc_auc:.4f})")