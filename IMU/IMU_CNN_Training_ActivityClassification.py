import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy import signal
import os
import glob
import sys
import random

# ==========================================
# 1. 設定區
# ==========================================
DATA_DIR = os.path.join('TrainingData', 'Static Sit')
WINDOW_SIZE = 128
STEP_SIZE = 64
FEATURES = [
    'accX', 'accY', 'accZ', 
    'gyroX', 'gyroY', 'gyroZ', 
    'roll', 'pitch', 'yaw'
]

# 每個正常視窗生成多少個異常視窗
AUGMENTATION_COUNT = 5 

# ==========================================
# 2. 資料生成邏輯 (核心修改區)
# ==========================================

def get_random_abnormal_factor():
    """
    隨機回傳異常倍率，包含：
    1. 停止呼吸 (0.0) -> 機率約 20%
    2. 過慢 (0.2 ~ 0.5) -> 機率約 40%
    3. 過快 (1.5 ~ 3.0) -> 機率約 40%
    """
    rand_val = random.random()
    
    if rand_val < 0.2:
        return 0.0  # 代表停止呼吸
    elif rand_val < 0.6:
        return random.uniform(0.2, 0.5)  # 過慢
    else:
        return random.uniform(1.5, 3.0)  # 過快

def generate_stop_sample(window_data):
    """
    模擬呼吸停止 (Apnea / Holding Breath)
    邏輯：取該視窗的平均姿勢(Mean)，填滿整個視窗，並加上微量感測器雜訊。
    """
    # 1. 取得當前姿勢的平均值 (1, 9)
    #    這樣能保留原本坐著的角度，只是沒了呼吸起伏
    posture_mean = np.mean(window_data, axis=0)
    
    # 2. 複製成 (128, 9) 的平坦訊號
    flat_signal = np.tile(posture_mean, (len(window_data), 1))
    
    # 3. 加上高斯白雜訊 (Gaussian Noise) 模擬真實感測器
    #    scale=0.02 視你的感測器靈敏度而定，通常微量即可
    noise = np.random.normal(loc=0.0, scale=0.02, size=flat_signal.shape)
    
    return flat_signal + noise

def generate_abnormal_sample(window_data, factor):
    """
    根據 factor 生成異常樣本
    """
    # --- 情況 A: 呼吸停止 (Factor == 0) ---
    if factor == 0.0:
        return generate_stop_sample(window_data)
    
    # --- 情況 B: 過快或過慢 (Resampling) ---
    original_len = len(window_data)
    num_features = window_data.shape[1]
    
    new_len = int(original_len / factor)
    
    # 防止極端情況導致 new_len 為 0
    if new_len == 0: new_len = 1
        
    resampled_data = signal.resample(window_data, new_len)
    
    output_data = np.zeros((original_len, num_features))
    
    if new_len < original_len:
        # 變快(變短)，重複填補
        repeat_times = (original_len // new_len) + 1
        tiled = np.tile(resampled_data, (repeat_times, 1))
        output_data = tiled[:original_len, :]
    else:
        # 變慢(變長)，截取
        output_data = resampled_data[:original_len, :]
        
    return output_data

def load_data_from_folder(folder_path):
    search_pattern = os.path.join(folder_path, '*.csv')
    file_list = glob.glob(search_pattern)
    
    if not file_list:
        print(f"錯誤：在 '{folder_path}' 找不到任何 .csv 檔案！")
        sys.exit(1)

    print(f"找到 {len(file_list)} 個檔案，開始處理 (包含呼吸停止 0 Hz 模擬)...")

    X_normal = []
    X_abnormal = []
    
    for file_path in file_list:
        try:
            df = pd.read_csv(file_path)
            if not all(col in df.columns for col in FEATURES): continue
            if len(df) < WINDOW_SIZE: continue
            
            data = df[FEATURES].values.astype(np.float32)
            
            # --- 製作正常樣本 (Class 0) ---
            for i in range(0, len(data) - WINDOW_SIZE, STEP_SIZE):
                window = data[i : i + WINDOW_SIZE]
                X_normal.append(window)
                
                # --- 製作異常樣本 (Class 1) ---
                for _ in range(AUGMENTATION_COUNT):
                    factor = get_random_abnormal_factor() # 這裡現在會回傳 0.0, 0.3, 2.5 等
                    abnormal_window = generate_abnormal_sample(window, factor)
                    X_abnormal.append(abnormal_window)

        except Exception as e:
            print(f"讀取錯誤 {file_path}: {e}")

    return np.array(X_normal), np.array(X_abnormal)

# ==========================================
# 3. 建立模型 (保持不變)
# ==========================================
def build_cnn_model(input_shape):
    model = models.Sequential([
        layers.Conv1D(32, 5, activation='relu', input_shape=input_shape, padding='same'),
        layers.MaxPooling1D(2),
        layers.BatchNormalization(),
        
        layers.Conv1D(64, 3, activation='relu', padding='same'),
        layers.MaxPooling1D(2),
        layers.BatchNormalization(),
        
        layers.Conv1D(128, 3, activation='relu', padding='same'),
        layers.GlobalAveragePooling1D(), 
        
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ==========================================
# 4. 主程式
# ==========================================
if __name__ == "__main__":
    # 1. 載入
    X_norm, X_abnorm = load_data_from_folder(DATA_DIR)
    
    if len(X_norm) == 0:
        print("無資料，結束。")
        sys.exit(1)
        
    print(f"正常樣本數: {len(X_norm)}")
    print(f"異常樣本數 (含停止/過快/過慢): {len(X_abnorm)}")
    
    # 2. 合併
    X = np.concatenate([X_norm, X_abnorm], axis=0)
    y = np.concatenate([np.zeros(len(X_norm)), np.ones(len(X_abnorm))], axis=0)
    
    # 3. 標準化
    N, T, F = X.shape
    X_flat = X.reshape(N * T, F)
    scaler = StandardScaler()
    X_flat_scaled = scaler.fit_transform(X_flat)
    X_scaled = X_flat_scaled.reshape(N, T, F)
    
    # 4. 切分
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 5. 訓練
    print("\n開始訓練...")
    model = build_cnn_model((WINDOW_SIZE, len(FEATURES)))
    model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_test, y_test))
    
    # 6. 存檔
    if not os.path.exists('models'): os.makedirs('models')
    model.save('models/breathing_cnn_model.h5')
    import joblib
    joblib.dump(scaler, 'models/scaler.pkl')
    
    print("\n完成！模型已學會區分：正常 vs (停止/過快/過慢)")