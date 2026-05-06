import os
import pandas as pd
import numpy as np
import tensorflow as tf
from scipy import signal
from tensorflow.keras.models import load_model
from layers_utils import PositionalEmbedding # Ensure this file is in your folder

# --- CONFIGURATION ---
BASE_DIR = "C:/Git/Python/IMU"
DATA_PATH = os.path.join(BASE_DIR, "data/Respiration Verfication Samples")

# Model File Paths
MODELS = {
    "CNN": os.path.join(BASE_DIR, "models/respiration_CNN_classifier.keras"),
    "CNN-Trans": os.path.join(BASE_DIR, "models/respiration_CNNxTran_classifier.keras"),
    "CNN-LSTM": os.path.join(BASE_DIR, "models/respiration_CNNxLSTM_classifier.keras")
}

# Target Files
TARGET_FILES = ["14 BPM.csv", "18 BPM.csv", "30 BPM - Straw.csv", "38 BPM.csv", "46 BPM.csv"]
FEATURES = ['accx', 'accy', 'accz', 'gyrox', 'gyroy', 'gyroz', 'roll', 'pitch', 'yaw']
WINDOW_SIZE = 200  # 10 seconds
FS = 20            # 20Hz

def apply_agc(sig, fs=20, window_sec=5):
    window_size = int(fs * window_sec)
    rolling_std = pd.Series(sig).rolling(window=window_size, center=True).std()
    rolling_std = rolling_std.bfill().ffill().replace(0, 1e-6)
    return sig / rolling_std.values

def clean_signal(data, fs=20):
    cleaned_data = np.zeros_like(data)
    nyq = 0.5 * fs
    b, a = signal.butter(3, [0.08/nyq, 0.8/nyq], btype='band')
    for i in range(data.shape[1]):
        feat_filt = signal.filtfilt(b, a, data[:, i])
        feat_agc = apply_agc(feat_filt, fs=fs)
        try:
            cleaned_data[:, i] = signal.savgol_filter(feat_agc, 15, 2)
        except:
            cleaned_data[:, i] = feat_agc
    return cleaned_data

def verify_models():
    print("-" * 60)
    print(" LOADING MODELS... ")
    print("-" * 60)
    
    loaded_models = {}
    for name, path in MODELS.items():
        if os.path.exists(path):
            # Load with custom objects (Transformers require PositionalEmbedding)
            loaded_models[name] = load_model(path, custom_objects={
                "PositionalEmbedding": PositionalEmbedding
            }, compile=False)
            print(f"Successfully loaded: {name}")
        else:
            print(f"MISSING MODEL: {name} at {path}")

    results_list = []

    print("\n" + "-" * 60)
    print(" PROCESSING SAMPLES... ")
    print("-" * 60)

    for file_name in TARGET_FILES:
        file_path = os.path.join(DATA_PATH, file_name)
        if not os.path.exists(file_path):
            print(f"Skipping: {file_name} (File not found)")
            continue

        # Load and handle timestamps
        df = pd.read_csv(file_path)
        df.columns = [c.lower() for c in df.columns]
        
        # Calculate Magnitudes (9 IMU axes -> 11 Features)
        acc_cols = [c for c in df.columns if 'acc' in c][:3]
        gyro_cols = [c for c in df.columns if 'gyro' in c][:3]
        acc_mag = np.linalg.norm(df[acc_cols].values, axis=1, keepdims=True)
        gyro_mag = np.linalg.norm(df[gyro_cols].values, axis=1, keepdims=True)
        
        raw_feat = df.reindex(columns=FEATURES).values.astype(np.float32)
        combined = np.hstack([raw_feat, acc_mag, gyro_mag])
        
        # Preprocessing (Filter + AGC)
        cleaned = clean_signal(combined, fs=FS)
        
        # Extract a stable 10-second window from the middle of the file
        if len(cleaned) >= WINDOW_SIZE:
            start = (len(cleaned) - WINDOW_SIZE) // 2
            window = cleaned[start : start + WINDOW_SIZE]
            
            # Per-window Z-score normalization (Crucial for CNN-Trans/CNN-LSTM)
            window_norm = (window - np.mean(window, axis=0)) / (np.std(window, axis=0) + 1e-6)
            input_tensor = window_norm.reshape(1, WINDOW_SIZE, 11)
            
            # Predict with each model
            prediction_row = {"Sample ID": file_name.replace(".csv", "")}
            for model_name, model in loaded_models.items():
                prob = float(model.predict(input_tensor, verbose=0)[0][0])
                # Logic: Normal <= 0.4, Abnormal > 0.4
                prediction_row[model_name] = "ABNORMAL" if prob > 0.4 else "NORMAL"
            
            results_list.append(prediction_row)

    # --- FINAL OUTPUT TABLE ---
    df_results = pd.DataFrame(results_list)
    
    # Ensure correct column order
    column_order = ["Sample ID", "CNN", "CNN-Trans", "CNN-LSTM"]
    df_results = df_results[[c for c in column_order if c in df_results.columns]]

    print("\n" + " COMPREHENSIVE PREDICTION REPORT ".center(70, "="))
    print(df_results.to_string(index=False))
    print("=" * 70)

if __name__ == "__main__":
    verify_models()