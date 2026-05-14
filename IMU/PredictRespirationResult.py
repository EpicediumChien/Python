import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from scipy import signal
import os
from collections import deque

# ==========================================
# 2. Configuration
# ==========================================
# FILE_PATH = os.path.join('data', 'StrawCompare', 'StaticSit_imu_YAHBOOM_20260320_002717.csv')
FILE_PATH = os.path.join('data', 'StrawCompare', 'Straw_Dyspnea_imu_YAHBOOM_20260320_003148.csv')
MODEL_PATH = 'models/respiration_classifier.keras'
SCALER_PATH = 'models/scaler.pkl'

TARGET_FREQ = 20        
WINDOW_SIZE = 200       
STEP_SIZE = 40          
FEATURES = ['accx', 'accy', 'accz', 'gyrox', 'gyroy', 'gyroz', 'roll', 'pitch', 'yaw']

BPM_MIN = 12.0
BPM_MAX = 20.0

# ==========================================
# 1. Custom Layer (Required for Loading)
# ==========================================
@tf.keras.utils.register_keras_serializable()
class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, sequence_length, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.embed_dim = embed_dim
        self.pos_emb = tf.keras.layers.Embedding(input_dim=sequence_length, output_dim=embed_dim)

    def call(self, inputs):
        length = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=length, delta=1)
        return inputs + self.pos_emb(positions)

    def get_config(self):
        config = super().get_config()
        config.update({"sequence_length": self.sequence_length, "embed_dim": self.embed_dim})
        return config

# ==========================================
# 2. Logic: Prediction Smoother
# ==========================================
class PredictionManager:
    """Stabilizes flickering BPM and AI confidence using temporal smoothing."""
    def __init__(self, size=5):
        self.bpm_history = deque(maxlen=size)
        self.prob_history = deque(maxlen=size)

    def update(self, bpm, prob):
        self.bpm_history.append(bpm)
        self.prob_history.append(prob)
        # Use median for BPM to ignore outliers; mean for probability
        return np.median(self.bpm_history), np.mean(self.prob_history)

# ==========================================
# 3. Enhanced Logic Functions
# ==========================================

def get_bpm_robust(window_data, fs=20):
    # Use Acc Mag (index 9)
    sig = window_data[:, 9] 
    # Check if the signal is 'dead' (sensor not moving)
    if np.std(sig) < 0.005: return 0.0
    
    sig = sig - np.mean(sig)
    n_fft = 2048 
    freqs = np.fft.rfftfreq(n_fft, d=1/fs)
    fft_mag = np.abs(np.fft.rfft(sig, n=n_fft))
    
    # Breathing range: 0.1Hz (6 BPM) to 0.75Hz (45 BPM)
    idx = np.where((freqs >= 0.1) & (freqs <= 0.75))[0]
    return freqs[idx][np.argmax(fft_mag[idx])] * 60

def apply_agc_logic(sig, fs=20, window_sec=5):
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
# 4. Main Execution
# ==========================================

if __name__ == "__main__":
    # --- Setup ---
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'PositionalEmbedding': PositionalEmbedding})
    scaler = joblib.load(SCALER_PATH)
    manager = PredictionManager(size=5) # Smooth over last 10 seconds of data

    # --- Data Loading ---
    df = pd.read_csv(FILE_PATH)
    df.columns = [c.lower().strip() for c in df.columns]
    
    # Standardize Timestamp and Resample
    time_col = 'unixtimestamp' if 'unixtimestamp' in df.columns else 'timestamp'
    df['dt'] = pd.to_datetime(df[time_col], unit='ms')
    df_res = df.set_index('dt').resample('50ms').mean().interpolate().ffill().bfill()

    # --- Feature Matrix ---
    acc_mag = np.linalg.norm(df_res[['accx', 'accy', 'accz']].values, axis=1, keepdims=True)
    gyro_mag = np.linalg.norm(df_res[['gyrox', 'gyroy', 'gyroz']].values, axis=1, keepdims=True)
    raw_feats = df_res.reindex(columns=FEATURES).values.astype(np.float32)
    combined = np.hstack([raw_feats, acc_mag, gyro_mag]) 
    
    cleaned = clean_signal_logic(combined, fs=TARGET_FREQ)

    print(f"\nFinal Hybrid Inference (20Hz Optimized):")
    print(f"{'Time':<8} | {'BPM':<6} | {'Status':<18} | {'Confidence'}")
    print("-" * 65)

    # --- Inference Loop ---
    AI_THRESHOLD = 0.70 # INCREASED threshold to reduce false abnormal pattern detections

    for i in range(0, len(cleaned) - WINDOW_SIZE, STEP_SIZE):
        window = cleaned[i : i + WINDOW_SIZE]
        
        # 1. Get raw metrics
        raw_bpm = get_bpm_robust(window, fs=TARGET_FREQ)
        window_scaled = scaler.transform(window.reshape(-1, 11)).reshape(1, WINDOW_SIZE, 11)
        raw_prob = model.predict(window_scaled, verbose=0)[0][0]
        
        # 2. Smooth metrics (Temporal stabilization)
        bpm, prob = manager.update(raw_bpm, raw_prob)
        
        # 3. Hybrid Classification Logic
        # Normal Range: 12 BPM (5s/breath) to 20 BPM (3s/breath)
        is_fast = bpm > 22.0
        is_slow = 0.5 < bpm < 11.0 
        is_pattern_bad = prob > AI_THRESHOLD 
        
        if is_fast:
            status = "ABNORMAL (Fast)"
        elif is_slow:
            status = "ABNORMAL (Slow)"
        elif is_pattern_bad:
            status = "ABNORMAL (Pattern)"
        else:
            status = "NORMAL"

        conf = prob if status.startswith("ABNORMAL") else (1 - prob)
        timestamp = i / TARGET_FREQ
        print(f"{timestamp:<8.1f} | {bpm:<6.1f} | {status:<18} | {conf:.1%}")