import os
import pandas as pd
import numpy as np
import tensorflow as tf
from scipy import signal
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
from layers_utils import PositionalEmbedding # 確保檔案在同目錄

# --- CONFIGURATION ---
BASE_DIR = "C:/Git/Python/IMU"
DATA_PATH = os.path.join(BASE_DIR, "data/Respiration Verfication Samples")

# 定義你要測試的所有模型
MODELS_TO_TEST = {
    "CNN_Basic": os.path.join(BASE_DIR, "models/respiration_CNN_classifier.keras"),
    "CNN_x_Transformer": os.path.join(BASE_DIR, "models/respiration_CNNxTran_classifier.keras"),
    "CNN_x_LSTM": os.path.join(BASE_DIR, "models/respiration_CNNxLSTM_classifier.keras")
}

FILES_TO_VERIFY = {
    "14 BPM.csv": "Normal",
    "30 BPM - Straw.csv": "Abnormal"
}

FEATURES = ['accx', 'accy', 'accz', 'gyrox', 'gyroy', 'gyroz', 'roll', 'pitch', 'yaw']
WINDOW_SIZE = 200
STEP_SIZE = 10
FS = 20

# --- PREPROCESSING ---
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

def run_multi_model_verification():
    # 設定 Pandas 顯示
    pd.set_option('display.max_rows', 10) # 預覽時不要顯示太多行，以免洗版
    pd.set_option('display.width', 1000)

    for model_name, model_path in MODELS_TO_TEST.items():
        print(f"\n" + "#"*60)
        print(f" TESTING MODEL: {model_name} ".center(60, '#'))
        print(f"#"*60)

        if not os.path.exists(model_path):
            print(f"Skipping... Model file not found: {model_path}")
            continue

        # 載入模型 (統一帶入 custom_objects)
        model = load_model(model_path, custom_objects={
            "PositionalEmbedding": PositionalEmbedding
        }, compile=False)
        
        all_results = []
        y_true, y_pred = [], []

        for file_name, truth_label in FILES_TO_VERIFY.items():
            path = os.path.join(DATA_PATH, file_name)
            if not os.path.exists(path): continue

            # 1. 讀取數據與特徵工程
            df = pd.read_csv(path)
            df.columns = [c.lower() for c in df.columns]
            acc_cols = [c for c in df.columns if 'acc' in c][:3]
            gyro_cols = [c for c in df.columns if 'gyro' in c][:3]
            acc_mag = np.linalg.norm(df[acc_cols].values, axis=1, keepdims=True)
            gyro_mag = np.linalg.norm(df[gyro_cols].values, axis=1, keepdims=True)
            raw_feat = df.reindex(columns=FEATURES).values.astype(np.float32)
            combined = np.hstack([raw_feat, acc_mag, gyro_mag])
            
            # 2. 清洗
            cleaned = clean_signal(combined, fs=FS)
            
            # 3. 滑動視窗
            for start_idx in range(0, len(cleaned) - WINDOW_SIZE + 1, STEP_SIZE):
                window = cleaned[start_idx : start_idx + WINDOW_SIZE]
                
                # 正規化 (Window Normalization)
                window_norm = (window - np.mean(window, axis=0)) / (np.std(window, axis=0) + 1e-6)
                
                input_tensor = window_norm.reshape(1, WINDOW_SIZE, 11)
                prob = float(model.predict(input_tensor, verbose=0)[0][0])
                
                # 判定邏輯 (統一使用 0.4)
                pred_label = "Abnormal" if prob > 0.4 else "Normal"
                
                all_results.append({
                    "Model": model_name,
                    "File": file_name,
                    "Truth": truth_label,
                    "Pred": pred_label,
                    "Prob": f"{prob:.4f}"
                })
                y_true.append(truth_label)
                y_pred.append(pred_label)

        # 每個模型跑完後輸出總結
        print(f"\n[{model_name}] Confusion Matrix:")
        labels = ["Abnormal", "Normal"]
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        print(pd.DataFrame(cm, index=[f"Act {l}" for l in labels], columns=[f"Pred {l}" for l in labels]))
        
        print(f"\n[{model_name}] Classification report:")
        print(classification_report(y_true, y_pred, labels=labels))

if __name__ == "__main__":
    run_multi_model_verification()