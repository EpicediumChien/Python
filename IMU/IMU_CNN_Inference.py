import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os

# ==========================================
# 1. 設定與載入
# ==========================================
MODEL_PATH = 'models/breathing_cnn_model.keras'
SCALER_PATH = 'models/scaler.pkl'
FEATURES = ['accX', 'accY', 'accZ', 'gyroX', 'gyroY', 'gyroZ', 'roll', 'pitch', 'yaw']
TARGET_FREQ = "40ms"  # 必須與訓練時一致 (25Hz)
WINDOW_SIZE = 150     # 必須與訓練時一致 (6秒)
STEP_SIZE = 25        # 預測時可以滑動快一點 (例如每 1 秒預測一次)

# 載入模型與標準化器
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    print("錯誤：找不到模型或標準化器檔案！")
    exit()

model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

def predict_file(file_path):
    print(f"\n正在分析檔案: {file_path}")
    
    # 1. 讀取與預處理 (與訓練邏輯完全相同)
    df = pd.read_csv(file_path)
    ts_col = [c for c in df.columns if 'timestamp' in c.lower()][0]
    df['datetime'] = pd.to_datetime(df[ts_col], unit='ms')
    df = df.sort_values('datetime').drop_duplicates(subset=['datetime']).set_index('datetime')
    
    # 重採樣至 25Hz
    df_resampled = df[FEATURES].resample(TARGET_FREQ).mean().interpolate().ffill().bfill()
    data = df_resampled.values.astype(np.float32)
    
    # 2. 標準化
    data_scaled = scaler.transform(data)
    
    # 3. 滑動窗口預測
    results = []
    timestamps = []
    
    for i in range(0, len(data_scaled) - WINDOW_SIZE, STEP_SIZE):
        window = data_scaled[i : i + WINDOW_SIZE]
        window = window.reshape(1, WINDOW_SIZE, len(FEATURES)) # 變成 (1, 150, 9)
        
        # 進行預測
        prob = model.predict(window, verbose=0)[0][0]
        label = "異常" if prob > 0.5 else "正常"
        
        # 紀錄時間點 (視窗的結尾時間)
        current_ts = df_resampled.index[i + WINDOW_SIZE]
        results.append((current_ts, label, prob))
        
        print(f"時間: {current_ts.strftime('%H:%M:%S')} | 狀態: {label} (機率: {prob:.2%})")

    return results

# ==========================================
# 4. 執行測試
# ==========================================
if __name__ == "__main__":
    # 你可以放一個你沒用來訓練的 CSV 檔案路徑
    test_csv = 'TrainingData/Static Sit/imu_log_20251210_222946.csv' 
    if os.path.exists(test_csv):
        analysis = predict_file(test_csv)
        
        # 簡單統計
        total = len(analysis)
        abnormal_count = sum(1 for _, label, _ in analysis if label == "異常")
        print(f"\n--- 分析完成 ---")
        print(f"總分析視窗數: {total}")
        print(f"異常警報次數: {abnormal_count}")
        print(f"異常比例: {abnormal_count/total:.2%}")
    else:
        print(f"請修改 test_csv 路徑以指向正確的檔案")