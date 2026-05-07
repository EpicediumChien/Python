import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, Bidirectional, Conv1D, MaxPooling1D, Dense, Dropout, BatchNormalization, Input, SpatialDropout1D, GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from scipy import signal
import os
import glob
import random
import joblib
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
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
STEP_SIZE = 10          
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
    savgol_win = 15 # 20Hz 下，Savgol 建議用 11 或 15 (約 0.5~0.7s)
    
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
    """Detects BPM using raw Accelerometer Z (usually strongest for sitting/lying)"""
    sig = window_data[:, 2] # AccZ
    sig = sig - np.mean(sig)
    freqs = np.fft.rfftfreq(len(sig), d=1/fs)
    fft_mag = np.abs(np.fft.rfft(sig))
    # Search a wider range to find the 30-46 BPM peaks
    idx = np.where((freqs >= 0.1) & (freqs <= 0.9))[0]
    if len(idx) == 0: return 0.0
    return freqs[idx][np.argmax(fft_mag[idx])] * 60

def augment_window(window):
    """Adds realistic sensor noise and amplitude variation"""
    # 1. Random Jitter (Sensor shaking)
    noise = np.random.normal(0, 0.005, window.shape)
    # 2. Random Amplitude Scaling (Deep vs Shallow)
    scale = random.uniform(0.8, 1.2)
    return (window + noise) * scale

def generate_abnormal_by_bpm(window_data, fs=20):
    orig_bpm = get_dominant_bpm(window_data, fs)
    orig_len = len(window_data)
    
    # Target Tachypnea (25-45) or Bradypnea (5-10)
    target_bpm = random.uniform(25, 45) if random.random() > 0.5 else random.uniform(5, 10)
    ratio = target_bpm / orig_bpm
    new_len = max(int(orig_len / ratio), 1)
    resampled = signal.resample(window_data, new_len)
    
    if ratio > 1.0:
        output = np.tile(resampled, (int(np.ceil(orig_len / new_len)), 1))[:orig_len, :]
    else:
        output = resampled[:orig_len, :]
        if len(output) < orig_len:
            output = np.pad(output, ((0, orig_len - len(output)), (0, 0)), mode='edge')
    
    return augment_window(output)

# ==========================================
# 4. Global Scaling Preparation
# ==========================================

def process_single_file(file_path):
    try:
        df = pd.read_csv(file_path)
        df.columns = [c.lower() for c in df.columns]
        
        # --- 1. SYNC TO 20Hz (Critical) ---
        # 50ms resampling ensures exactly 20Hz. 
        # 40ms was creating 25Hz and ruining your BPM math.
        ts_col = next(c for c in ['unix_timestamp', 'timestamp'] if c in df.columns)
        df['dt'] = pd.to_datetime(df[ts_col], unit='ms')
        df_res = df.set_index('dt').resample('50ms').mean().interpolate().ffill().bfill()

        # --- 2. FEATURE ENGINEERING ---
        acc_cols = [c for c in df_res.columns if 'acc' in c][:3]
        gyro_cols = [c for c in df_res.columns if 'gyro' in c][:3]
        acc_mag = np.linalg.norm(df_res[acc_cols].values, axis=1, keepdims=True)
        gyro_mag = np.linalg.norm(df_res[gyro_cols].values, axis=1, keepdims=True)
        raw_data = df_res.reindex(columns=[f.lower() for f in FEATURES]).values.astype(np.float32)
        combined = np.hstack([raw_data, acc_mag, gyro_mag]) 
        
        # --- 3. CLEANING ---
        processed_data = clean_signal_logic(combined, fs=TARGET_FREQ)

        x_list, y_list = [], []
        for i in range(0, len(processed_data) - WINDOW_SIZE, STEP_SIZE):
            window = processed_data[i : i + WINDOW_SIZE]
            bpm = get_dominant_bpm(window, fs=TARGET_FREQ)
            
            # --- 回歸 Z-score 正規化 ---
            # 必須除以標準差，模型才能看清不同感測器軸的「形狀」
            window_norm = (window - np.mean(window, axis=0)) / (np.std(window, axis=0) + 1e-6)

            if 10.0 <= bpm <= 22.0:
                # LABEL 0: NORMAL
                x_list.append(window_norm)
                y_list.append(0) 
                
                # --- 5. AUGMENTATION ---
                # Create synthetic abnormal data from normal samples
                for _ in range(AUGMENTATION_COUNT):
                    ab_win = generate_abnormal_by_bpm(window, fs=TARGET_FREQ)
                    # 增強數據也必須進行 Z-score
                    ab_norm = (ab_win - np.mean(ab_win, axis=0)) / (np.std(ab_win, axis=0) + 1e-6)
                    x_list.append(ab_norm)
                    y_list.append(1) 
            else:
                # LABEL 1: ABNORMAL (Naturally occurring)
                x_list.append(window_norm)
                y_list.append(1)
                
        return np.array(x_list), np.array(y_list)
    except Exception as e:
        print(f"Error in {file_path}: {e}")
        return None, None

# ==========================================
# 4. Model & Training
# ==========================================
def build_model(input_shape):
    inputs = Input(shape=input_shape)
    x = GaussianNoise(0.02)(inputs) 

    # 1. Wider CNN Kernels (Size 15) to see the breath wave shape
    x = Conv1D(64, 15, padding='same', activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)

    # 2. Bi-LSTM with internal dropout to prevent memorizing noise
    x = Bidirectional(LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.2))(x)
    
    # 3. Pool the rhythm features
    avg_p = GlobalAveragePooling1D()(x)
    max_p = GlobalMaxPooling1D()(x)
    combined = concatenate([avg_p, max_p])

    # 4. Fully Connected
    x = Dense(32, activation='relu')(combined)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), 
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    file_list = glob.glob(os.path.join(DATA_DIR, '*.csv'))
    random.shuffle(file_list)
    
    def collect_and_balance(files):
        x_raw, y_raw = [], []
        for f in files:
            xf, yf = process_single_file(f)
            if xf is not None:
                x_raw.append(xf)
                y_raw.append(yf)
        
        X = np.concatenate(x_raw)
        y = np.concatenate(y_raw)
        
        # --- Hard Balancing Logic ---
        idx_0 = np.where(y == 0)[0]
        idx_1 = np.where(y == 1)[0]
        
        # Downsample majority class to match minority class
        min_samples = min(len(idx_0), len(idx_1))
        np.random.shuffle(idx_0)
        np.random.shuffle(idx_1)
        
        balanced_idx = np.concatenate([idx_0[:min_samples], idx_1[:min_samples]])
        np.random.shuffle(balanced_idx)
        
        return X[balanced_idx], y[balanced_idx]

    print("Processing and Balancing Dataset...")
    split = int(len(file_list) * 0.8)
    X_train, y_train = collect_and_balance(file_list[:split])
    X_test, y_test = collect_and_balance(file_list[split:])

    print(f"Final Training Balance: {np.bincount(y_train.astype(int))}")

    model = build_model((WINDOW_SIZE, X_train.shape[2]))
    
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=64,
        validation_data=(X_test, y_test),
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
        ]
    )

    # ==========================================
    # 5. Evaluation & Results Visualization
    # ==========================================
    print("\n--- Generating Training Results ---")

    # 1. Predict on Test Set
    y_pred_prob = model.predict(X_test)
    y_pred_tuned = (y_pred_prob > 0.4).astype(int) 

    # 2. Classification Report
    print("\n[Classification Report]")
    target_names = ['Normal (Eupnea)', 'Abnormal (Dyspnea/Apnea)']
    print(classification_report(y_test, y_pred_tuned, target_names=target_names))

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
    cm = confusion_matrix(y_test, y_pred_tuned)
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
    model.save('models/respiration_CNNxLSTM_classifier.keras')
    # joblib.dump(scaler, 'models/CNNxLSTM_scaler.pkl')
    print(f"\nTraining Complete. Model saved to 'models/respiration_CNNxLSTM_classifier.keras' (AUC: {roc_auc:.4f})")