import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from scipy import signal
import os
import glob
import sys
import random
import joblib

# ==========================================
# 1. 設定區
# ==========================================
DATA_DIR = os.path.join('TrainingData', 'Static Sit')
TARGET_FREQ = "40ms"    # 25Hz
WINDOW_SIZE = 150       # 6秒
STEP_SIZE = 50          # 步長
FEATURES = ['accX', 'accY', 'accZ', 'gyroX', 'gyroY', 'gyroZ', 'roll', 'pitch', 'yaw']
AUGMENTATION_COUNT = 3  

# ==========================================
# 2. 資料處理邏輯 (按檔案讀取)
# ==========================================

def get_random_abnormal_factor():
    rand_val = random.random()
    if rand_val < 0.3: return 0.0 # 停止
    elif rand_val < 0.65: return random.uniform(0.3, 0.5) # 慢
    else: return random.uniform(2.0, 3.5) # 快

def generate_abnormal_sample(window_data, factor):
    if factor == 0.0:
        posture_mean = np.mean(window_data, axis=0)
        return np.tile(posture_mean, (len(window_data), 1)) + np.random.normal(0, 0.02, window_data.shape)
    
    orig_len = len(window_data)
    new_len = max(int(orig_len / factor), 1)
    resampled = signal.resample(window_data, new_len)
    output = np.tile(resampled, (orig_len // new_len + 1, 1))[:orig_len, :]
    return output + np.random.normal(0, 0.005, output.shape)

def process_single_file(file_path):
    """處理單個檔案並回傳視窗化後的 X, y"""
    try:
        df = pd.read_csv(file_path)
        ts_col = [c for c in df.columns if 'timestamp' in c.lower()]
        if not ts_col: return None, None
        
        df['datetime'] = pd.to_datetime(df[ts_col[0]], unit='ms')
        df = df.sort_values('datetime').drop_duplicates(subset=['datetime']).set_index('datetime')
        df_resampled = df[FEATURES].resample(TARGET_FREQ).mean().interpolate().ffill().bfill()
        data = df_resampled.values.astype(np.float32)

        x_list, y_list = [], []
        if len(data) < WINDOW_SIZE: return None, None
        
        for i in range(0, len(data) - WINDOW_SIZE, STEP_SIZE):
            window = data[i : i + WINDOW_SIZE]
            x_list.append(window)
            y_list.append(0) # 正常
            
            for _ in range(AUGMENTATION_COUNT):
                x_list.append(generate_abnormal_sample(window, get_random_abnormal_factor()))
                y_list.append(1) # 異常
        return np.array(x_list), np.array(y_list)
    except:
        return None, None

# ==========================================
# 3. 模型結構
# ==========================================
def build_model(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.GaussianNoise(0.01),
        layers.Conv1D(32, 7, activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.005)),
        layers.SpatialDropout1D(0.2),
        layers.MaxPooling1D(2),
        layers.BatchNormalization(),
        layers.Conv1D(64, 5, activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.005)),
        layers.SpatialDropout1D(0.2),
        layers.MaxPooling1D(2),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(), 
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    )
    return model

# ==========================================
# 4. 主程式
# ==========================================
if __name__ == "__main__":
    file_list = glob.glob(os.path.join(DATA_DIR, '*.csv'))
    random.shuffle(file_list)
    
    # --- 關鍵修正：按檔案切分 (80% 訓練, 20% 驗證) ---
    split_idx = int(len(file_list) * 0.8)
    train_files = file_list[:split_idx]
    test_files = file_list[split_idx:]
    
    def collect_data(files):
        x_all, y_all = [], []
        for f in files:
            x, y = process_single_file(f)
            if x is not None:
                x_all.append(x); y_all.append(y)
        return np.concatenate(x_all), np.concatenate(y_all)

    print("正在準備訓練資料...")
    X_train_raw, y_train = collect_data(train_files)
    print("正在準備驗證資料...")
    X_test_raw, y_test = collect_data(test_files)

    # 標準化
    scaler = StandardScaler()
    N, T, F = X_train_raw.shape
    X_train = np.nan_to_num(scaler.fit_transform(X_train_raw.reshape(-1, F))).reshape(N, T, F)
    
    N_t, T_t, F_t = X_test_raw.shape
    X_test = np.nan_to_num(scaler.transform(X_test_raw.reshape(-1, F_t))).reshape(N_t, T_t, F_t)

    # 權重
    weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(weights))

    model = build_model((WINDOW_SIZE, len(FEATURES)))
    model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        class_weight=class_weight_dict,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
        ]
    )

    if not os.path.exists('models'): os.makedirs('models')
    model.save('models/breathing_cnn_model.keras')
    joblib.dump(scaler, 'models/scaler.pkl')
    print("\n[完成] 已修正資料洩漏問題並完成訓練。")