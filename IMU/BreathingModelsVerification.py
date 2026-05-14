import os
import pandas as pd
import numpy as np
import tensorflow as tf
from scipy import signal
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
from layers_utils import PositionalEmbedding 

# --- 配置 ---
BASE_DIR = "C:/Git/Python/IMU"
DATA_PATH = os.path.join(BASE_DIR, "data/Respiration Verfication Samples")
MODELS_TO_TEST = {
    "CNN": os.path.join(BASE_DIR, "models/respiration_CNN_classifier.keras"),
    "Trans": os.path.join(BASE_DIR, "models/respiration_CNNxTran_classifier.keras"),
    "LSTM": os.path.join(BASE_DIR, "models/respiration_CNNxLSTM_classifier.keras")
}
FILES_TO_VERIFY = {"14 BPM.csv": "Normal", "30 BPM - Straw.csv": "Abnormal"}
FEATURES = ['accX', 'accY', 'accZ', 'gyroX', 'gyroY', 'gyroZ', 'roll', 'pitch', 'yaw']
WINDOW_SIZE, STEP_SIZE, FS = 200, 10, 20

# --- 2. 訊號處理 (精確複製訓練腳本邏輯) ---
def apply_agc_logic(sig, fs=20, window_sec=10):
    window_size = int(fs * window_sec)
    rolling_std = pd.Series(sig).rolling(window=window_size, center=True).std()
    rolling_std = rolling_std.bfill().ffill().replace(0, 1e-6)
    return sig / rolling_std.values

def clean_signal_logic(data, fs=20):
    """這段代碼必須與訓練腳本的 clean_signal_logic 完全相同"""
    cleaned_data = np.zeros_like(data)
    nyq = 0.5 * fs
    # 訓練腳本使用的是：2階, [0.08, 0.85]
    b, a = signal.butter(2, [0.08/nyq, 0.85/nyq], btype='band')
    for i in range(data.shape[1]):
        feat_filt = signal.filtfilt(b, a, data[:, i])
        # 訓練腳本第42行：window_sec=10
        feat_agc = apply_agc_logic(feat_filt, fs=fs, window_sec=10)
        # 訓練腳本第46行：savgol_filter window_length=savgol_win (15)
        try:
            cleaned_data[:, i] = signal.savgol_filter(feat_agc, 15, 2)
        except:
            cleaned_data[:, i] = feat_agc
    return cleaned_data

def run_diagnostic():
    for model_name, model_path in MODELS_TO_TEST.items():
        print(f"\n" + "="*70)
        print(f" DIAGNOSING MODEL: {model_name} ".center(70, '='))
        print(f"="*70)

        if not os.path.exists(model_path): continue
        model = load_model(model_path, custom_objects={"PositionalEmbedding": PositionalEmbedding}, compile=False)
        
        file_probs = {}

        for file_name, truth_label in FILES_TO_VERIFY.items():
            path = os.path.join(DATA_PATH, file_name)
            df = pd.read_csv(path)
            df.columns = [c.lower() for c in df.columns]
            
            acc_cols = [c for c in df.columns if 'acc' in c][:3]
            gyro_cols = [c for c in df.columns if 'gyro' in c][:3]
            m1 = np.linalg.norm(df[acc_cols].values, axis=1, keepdims=True)
            m2 = np.linalg.norm(df[gyro_cols].values, axis=1, keepdims=True)
            raw_feat = df.reindex(columns=[f.lower() for f in FEATURES]).values.astype(np.float32)
            combined = np.hstack([raw_feat, m1, m2])
            
            cleaned = clean_signal_logic(combined, fs=FS)
            
            probs = []
            for start_idx in range(0, len(cleaned) - WINDOW_SIZE + 1, STEP_SIZE):
                window = cleaned[start_idx : start_idx + WINDOW_SIZE]
                window_norm = (window - np.mean(window, axis=0)) / (np.std(window, axis=0) + 1e-6)
                prob = float(model.predict(window_norm.reshape(1, 200, 11), verbose=0)[0][0])
                probs.append(prob)
            
            # --- 關鍵：執行平滑處理 (每 5 個視窗取平均) ---
            smoothed_probs = pd.Series(probs).rolling(window=5, center=True).mean().bfill().ffill().tolist()
            file_probs[file_name] = smoothed_probs
            # print(f"File: {file_name:<18} | Avg Prob: {np.mean(probs):.4f} | Max: {np.max(probs):.4f} | Min: {np.min(probs):.4f}")

        # --- 根據平均值找出建議的 Threshold ---
        avg_14bpm = np.mean(file_probs["14 BPM.csv"])
        avg_30bpm = np.mean(file_probs["30 BPM - Straw.csv"])
        suggested_threshold = (avg_14bpm + avg_30bpm) / 2
        # --- 論文等級的門檻校準 ---
        if "LSTM" in model_name:
            current_threshold = 0.0367  # LSTM 的黃金分割點
        elif "CNN" in model_name:
            current_threshold = 0.3555  # CNN 勉強可以區分
        else: # Transformer
            current_threshold = 0.0892  # Transformer 在此數據集上表現較差

        print(f"\n>>> Suggested Threshold for {model_name}: {suggested_threshold:.4f}")
        
        # --- 使用建議門檻產出混淆矩陣 ---
        y_true, y_pred = [], []
        for file_name, truth_label in FILES_TO_VERIFY.items():
            for p in file_probs[file_name]:
                y_true.append(truth_label)
                # 如果機率大於建議門檻，判定為 Abnormal
                y_pred.append("Abnormal" if p > suggested_threshold else "Normal")

        labels = ["Abnormal", "Normal"]
        print(f"\n[{model_name}] Confusion Matrix (using {suggested_threshold:.4f}):")
        print(pd.DataFrame(confusion_matrix(y_true, y_pred, labels=labels), 
                           index=[f"Act {l}" for l in labels], columns=[f"Pred {l}" for l in labels]))

if __name__ == "__main__":
    run_diagnostic()