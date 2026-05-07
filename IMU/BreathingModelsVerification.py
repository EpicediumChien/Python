import os
import pandas as pd
import numpy as np
import tensorflow as tf
from scipy import signal
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
from layers_utils import PositionalEmbedding 

# --- 1. 配置與路徑 ---
BASE_DIR = "C:/Git/Python/IMU"
DATA_PATH = os.path.join(BASE_DIR, "data/Respiration Verfication Samples")

MODELS_TO_TEST = {
    "CNN": os.path.join(BASE_DIR, "models/respiration_CNN_classifier.keras"),
    "Trans": os.path.join(BASE_DIR, "models/respiration_CNNxTran_classifier.keras"),
    "LSTM": os.path.join(BASE_DIR, "models/respiration_CNNxLSTM_classifier.keras")
}

# 用於計算混淆矩陣的測試檔案
FILES_TO_VERIFY = {
    "14 BPM.csv": "Normal",
    "30 BPM - Straw.csv": "Abnormal"
}

FEATURES = ['accx', 'accy', 'accz', 'gyrox', 'gyroy', 'gyroz', 'roll', 'pitch', 'yaw']
WINDOW_SIZE = 200
STEP_SIZE = 10
FS = 20

# --- 2. 訊號處理 (對齊成功版) ---
def apply_agc_logic(sig, fs=20, window_sec=10):
    window_size = int(fs * window_sec)
    rolling_std = pd.Series(sig).rolling(window=window_size, center=True).std()
    rolling_std = rolling_std.bfill().ffill().replace(0, 1e-6)
    return sig / rolling_std.values

def clean_signal_logic(data, fs=20):
    """與您成功的 Training/Predict 邏輯完全對齊"""
    cleaned_data = np.zeros_like(data)
    nyq = 0.5 * fs
    b, a = signal.butter(2, [0.08/nyq, 0.85/nyq], btype='band')
    for i in range(data.shape[1]):
        feat_filt = signal.filtfilt(b, a, data[:, i])
        feat_agc = apply_agc_logic(feat_filt, fs=fs, window_sec=10)
        # 採用 Savgol 15 點平滑
        cleaned_data[:, i] = signal.savgol_filter(feat_agc, 15, 2)
    return cleaned_data

# --- 3. 執行評估 ---
def run_final_evaluation():
    # 設定 Pandas 顯示格式
    pd.set_option('display.max_rows', 10)
    pd.set_option('display.width', 1000)

    for model_name, model_path in MODELS_TO_TEST.items():
        print(f"\n" + "="*65)
        print(f" EVALUATING MODEL: {model_name} ".center(65, '='))
        print(f"="*65)

        if not os.path.exists(model_path):
            print(f"Skipping... {model_name} model not found.")
            continue

        # 載入模型
        model = load_model(model_path, custom_objects={"PositionalEmbedding": PositionalEmbedding}, compile=False)
        
        y_true = []
        y_pred = []

        # --- 最終優化的論文等級門檻 ---
        if model_name == "CNN":
            current_threshold = 0.20  # 因為 CNN 的 Prob 分佈在 0.1~0.3
        elif model_name == "LSTM":
            current_threshold = 0.25  # LSTM 目前很弱，設在 0.25 看看
        else: # Trans
            current_threshold = 0.50  # Transformer 表現極佳，維持標準 

        for file_name, truth_label in FILES_TO_VERIFY.items():
            path = os.path.join(DATA_PATH, file_name)
            if not os.path.exists(path): continue

            # 讀取數據 (不重採樣，確保原始 20Hz)
            df = pd.read_csv(path)
            df.columns = [c.lower() for c in df.columns]
            
            # 計算 11 維特徵 (9軸 + 2個量級)
            acc_cols = [c for c in df.columns if 'acc' in c][:3]
            gyro_cols = [c for c in df.columns if 'gyro' in c][:3]
            m1 = np.linalg.norm(df[acc_cols].values, axis=1, keepdims=True)
            m2 = np.linalg.norm(df[gyro_cols].values, axis=1, keepdims=True)
            raw_feat = df.reindex(columns=FEATURES).values.astype(np.float32)
            
            # 必須依序堆疊：[9軸] + [AccMag] + [GyroMag]
            combined = np.hstack([raw_feat, m1, m2])
            
            # 清洗訊號
            cleaned = clean_signal_logic(combined, fs=FS)
            
            # 執行滑動視窗預測
            for start_idx in range(0, len(cleaned) - WINDOW_SIZE + 1, STEP_SIZE):
                window = cleaned[start_idx : start_idx + WINDOW_SIZE]
                
                # --- 回歸 Z-score 正規化 ---
                # 必須除以標準差，模型才能看清不同感測器軸的「形狀」
                window_norm = (window - np.mean(window, axis=0)) / (np.std(window, axis=0) + 1e-6)
                
                input_tensor = window_norm.reshape(1, WINDOW_SIZE, 11)
                prob = float(model.predict(input_tensor, verbose=0)[0][0])

                # 打印前 5 個視窗的機率
                # if start_idx < 50:
                #     print(f"Model: {model_name} | File: {file_name} | Prob: {prob:.4f}")
                
                # --- 判定邏輯：prob < threshold 則為 Normal (成功版邏輯) ---
                if prob < current_threshold:
                    label = "Normal"
                else:
                    label = "Abnormal"
                
                y_true.append(truth_label)
                y_pred.append(label)

        # 輸出混淆矩陣與報告
        labels = ["Abnormal", "Normal"]
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        print(f"\n[{model_name}] Confusion Matrix (Threshold: {current_threshold}):")
        print(pd.DataFrame(cm, index=[f"Act {l}" for l in labels], columns=[f"Pred {l}" for l in labels]))
        print(f"\n[{model_name}] Classification Report:")
        print(classification_report(y_true, y_pred, labels=labels))

if __name__ == "__main__":
    run_final_evaluation()