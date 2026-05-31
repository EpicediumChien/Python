import os
import pandas as pd
import numpy as np
import tensorflow as tf
from scipy import signal
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
from layers_utils import PositionalEmbedding 
import matplotlib.pyplot as plt # 新增繪圖引用
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# --- 配置 ---
BASE_DIR = "C:/Git/Python/IMU"
DATA_PATH = os.path.join(BASE_DIR, "data/Respiration Verfication Samples")
MODELS_TO_TEST = {
    "CNN": os.path.join(BASE_DIR, "models/respiration_CNN_classifier.keras"),
    "Trans": os.path.join(BASE_DIR, "models/respiration_CNNxTran_classifier.keras"),
    "LSTM": os.path.join(BASE_DIR, "models/respiration_CNNxLSTM_classifier.keras")
}
# "Slow Walk Verify"
FILES_TO_VERIFY = {"14 BPM.csv": "Normal", "30 BPM - Straw.csv": "Abnormal"}
# FILES_TO_VERIFY = {"Slow Walk Verify.csv": "Normal", "30 BPM - Straw.csv": "Abnormal"}
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
    # 設置繪圖風格
    plt.style.use('seaborn-v0_8-whitegrid')
    
    for model_name, model_path in MODELS_TO_TEST.items():
        print(f"\n" + "="*70)
        print(f" DIAGNOSING MODEL: {model_name} ".center(70, '='))
        print(f"="*70)

        if not os.path.exists(model_path): continue
        model = load_model(model_path, custom_objects={"PositionalEmbedding": PositionalEmbedding}, compile=False)
        
        file_probs = {}
        all_y_true_binary = []  # 用於 ROC 的數值化標籤
        all_y_scores = []       # 模型輸出的機率

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
            
            # 執行平滑處理
            smoothed_probs = pd.Series(probs).rolling(window=5, center=True).mean().bfill().ffill().tolist()
            file_probs[file_name] = smoothed_probs
            
            # 收集用於 ROC 的數據
            label_int = 1 if truth_label == "Abnormal" else 0
            for p in smoothed_probs:
                all_y_true_binary.append(label_int)
                all_y_scores.append(p)

        # --- 門檻計算 ---
        # avg_normal = np.mean(file_probs["Slow Walk Verify.csv"])
        avg_normal = np.mean(file_probs["14 BPM.csv"])
        avg_abnormal = np.mean(file_probs["30 BPM - Straw.csv"])
        suggested_threshold = (avg_normal + avg_abnormal) / 2

        print(f"\n>>> Suggested Threshold for {model_name}: {suggested_threshold:.4f}")
        
        # --- 產出混淆矩陣 (基於建議門檻) ---
        y_pred_labels = ["Abnormal" if p > suggested_threshold else "Normal" for p in all_y_scores]
        y_true_labels = ["Abnormal" if i == 1 else "Normal" for i in all_y_true_binary]

        labels = ["Abnormal", "Normal"]
        cm = confusion_matrix(y_true_labels, y_pred_labels, labels=labels)
        print(f"\n[{model_name}] Confusion Matrix:")
        print(pd.DataFrame(cm, index=[f"Act {l}" for l in labels], columns=[f"Pred {l}" for l in labels]))

        # --- 繪製 ROC 曲線 ---
        fpr, tpr, thresholds = roc_curve(all_y_true_binary, all_y_scores)
        roc_auc = auc(fpr, tpr)

        # 計算當前門檻在 ROC 圖上的位置
        current_tpr = cm[0, 0] / (cm[0, 0] + cm[0, 1]) # TP / (TP + FN)
        current_fpr = cm[1, 0] / (cm[1, 0] + cm[1, 1]) # FP / (FP + TN)

        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        
        # 標註建議門檻點
        plt.scatter(current_fpr, current_tpr, color='red', s=80, label=f'Suggested Threshold ({suggested_threshold:.4f})', zorder=5)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.show() # 這會一個模型顯示一張圖，關閉圖表後會顯示下一個模型

if __name__ == "__main__":
    run_diagnostic()