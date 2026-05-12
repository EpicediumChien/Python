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
from tensorflow.keras.layers import (GaussianNoise, Input, Conv1D, BatchNormalization, MaxPooling1D, 
                                    GlobalAveragePooling1D, GlobalMaxPooling1D, 
                                    Dense, Dropout, concatenate, SpatialDropout1D)

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

    # --- 關鍵修正：模擬吸管呼吸/呼吸困難的物理特徵 ---

    # 非線性波形扭曲 (Simulate Effort/Sharpness)
    # 透過次方運算讓波峰變尖，模擬吸氣吃力（Gasping）的波形
    gamma = random.uniform(1.2, 1.5) 
    output = np.sign(output) * (np.abs(output) ** gamma)

    # 模擬肌肉震顫雜訊 (Muscle Tremor/Jitter)
    # 呼吸困難時身體會有細微抖動，這在 IMU 上是重要的 Abnormal 特徵
    output += np.random.normal(0, 0.015, output.shape)
    
    return augment_window(output)

# ==========================================
# 4. Global Scaling Preparation
# ==========================================

def process_single_file(file_path):
    try:
        df = pd.read_csv(file_path)
        df.columns = [c.lower() for c in df.columns]
        if len(df) < 50: # 原始數據太少，直接跳過
            return None, None
        
        # --- 1. SYNC TO 20Hz (Critical) ---
        # 50ms resampling ensures exactly 20Hz. 
        # 40ms was creating 25Hz and ruining your BPM math.
        ts_col = next(c for c in ['unix_timestamp', 'timestamp'] if c in df.columns)
        df['dt'] = pd.to_datetime(df[ts_col], unit='ms')
        df_res = df.set_index('dt').resample('50ms').mean().interpolate().ffill().bfill()
        
        # 關鍵修正：檢查重採樣後的長度是否足以進行濾波 (Savgol_win=15) 與 窗口切分
        if len(df_res) <= 20: # 至少要比 savgol_win 大
            return None, None

        # --- 2. FEATURE ENGINEERING ---
        acc_cols = [c for c in df_res.columns if 'acc' in c][:3]
        gyro_cols = [c for c in df_res.columns if 'gyro' in c][:3]
        acc_mag = np.linalg.norm(df_res[acc_cols].values, axis=1, keepdims=True)
        gyro_mag = np.linalg.norm(df_res[gyro_cols].values, axis=1, keepdims=True)
        raw_data = df_res.reindex(columns=[f.lower() for f in FEATURES]).values.astype(np.float32)
        combined = np.hstack([raw_data, acc_mag, gyro_mag]) 
        
        # --- 3. CLEANING ---
        processed_data = clean_signal_logic(combined, fs=TARGET_FREQ)

        # 檢查清理後是否有足夠長度切出至少一個 Window
        if len(processed_data) < WINDOW_SIZE:
            return None, None

        x_list, y_list = [], []
        for i in range(0, len(processed_data) - WINDOW_SIZE, STEP_SIZE):
            window = processed_data[i : i + WINDOW_SIZE]
            bpm = get_dominant_bpm(window, fs=TARGET_FREQ)
            
            # --- 核心修正 1：對所有數據（包含 Normal）加入微量底噪 ---
            # 這能防止模型把「感測器雜訊」誤認為「呼吸異常」
            window = window + np.random.normal(0, 0.002, window.shape)
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
# 1. FIXED MODEL: Added L2, Noise, and Hybrid Pooling
def build_model(input_shape):
    inputs = Input(shape=input_shape)
    
    # 增加微量高斯雜訊，增加模型穩健性
    x = GaussianNoise(0.02)(inputs)

    # 第一層：大核捲積 (Size 15) - 捕捉呼吸的主頻率波形
    # 在 20Hz 下，15 點約為 0.75s，能涵蓋半個呼吸相位
    x1 = Conv1D(64, 15, padding='same', activation='relu', kernel_regularizer=l2(1e-3))(x)
    x1 = BatchNormalization()(x1)
    x1 = SpatialDropout1D(0.4)(x1) # 隨機丟棄整個通道，強迫模型學習多個軸
    x1 = Dropout(0.3)(x1)

    # 第二層：中核捲積 (Size 7) - 捕捉呼吸深度變化的特徵
    x2 = Conv1D(128, 7, padding='same', activation='relu', kernel_regularizer=l2(1e-3))(x1)
    x2 = BatchNormalization()(x2)
    x2 = MaxPooling1D(2)(x2) # 200 -> 100
    x2 = Dropout(0.3)(x2)

    # 第三層：小核捲積 (Size 3) - 捕捉吸管呼吸或異常導致的高頻細微震顫
    x3 = Conv1D(128, 3, padding='same', activation='relu')(x2)
    x3 = BatchNormalization()(x3)
    x3 = Dropout(0.3)(x3)
    
    # 關鍵：混合池化 (Hybrid Pooling)
    # GlobalAveragePooling 捕捉整體節奏，GlobalMaxPooling 捕捉異常的突發信號
    avg_p = GlobalAveragePooling1D()(x3)
    max_p = GlobalMaxPooling1D()(x3)
    combined = concatenate([avg_p, max_p])

    # 全連接層
    x = Dense(64, activation='relu')(combined)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="Respiration_CNN")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4), # 稍微調高學習率
        loss='binary_crossentropy', # 論文建議先用標準 BCELoss
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
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
        epochs=150,
        batch_size=32, # 較小的 Batch size 對 CNN 尋找細節特徵有幫助
        validation_data=(X_test, y_test),
        callbacks=[
            EarlyStopping(patience=10, restore_best_weights=True, monitor='val_auc', mode='max'),
            ReduceLROnPlateau(patience=8, factor=0.5, monitor='val_loss')
        ]
    )

    # ==========================================
    # 5. Evaluation & Results Visualization
    # ==========================================
    print("\n--- Generating Training Results ---")

    target_names = ['Normal (Eupnea)', 'Abnormal (Dyspnea/Apnea)']

    # Results Visualization (Standardized)
    y_pred_prob = model.predict(X_test).ravel()
    
    # 計算 ROC 曲線並找出最佳閾值 (Youden's Index)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    print(f"\n>>> Optimal Threshold based on ROC: {optimal_threshold:.4f}")

    y_pred = (y_pred_prob >= optimal_threshold).astype(int)
    # 顯示分類報告
    print("\n[Final Classification Report]")
    print(classification_report(y_test, y_pred, target_names=target_names))

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