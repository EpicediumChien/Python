import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

# 設定
VERIFY_DATA_DIR = os.path.join('data', 'Movement Verify Samples')
FS = 20  # 採樣率 20Hz
TARGET_LABELS = ['Slow Walk', 'Fast Walk']

def clean_signal_logic(data, fs=20):
    """使用者提供的預處理邏輯"""
    cleaned_data = np.zeros_like(data)
    nyq = 0.5 * fs 
    b, a = signal.butter(2, [0.1/nyq, 9.0/nyq], btype='band')
    for i in range(data.shape[1]):
        feat_filt = signal.filtfilt(b, a, data[:, i])
        try:
            cleaned_data[:, i] = signal.savgol_filter(feat_filt, window_length=11, polyorder=2)
        except:
            cleaned_data[:, i] = feat_filt
    return cleaned_data

def get_data_and_time(label):
    """讀取、修正欄位、計算量級並生成時間軸"""
    dir_path = os.path.join(VERIFY_DATA_DIR, label)
    if not os.path.exists(dir_path): return None
    files = [f for f in os.listdir(dir_path) if f.endswith('.csv')]
    if not files: return None
    
    # 讀取第一個檔案
    df = pd.read_csv(os.path.join(dir_path, files[0]))
    
    # 根據範例：真正的加速度在 [AccX_Real, AccY, AccZ] 對應索引 [3, 4, 5]
    # 因為欄位 2 被重複的時間戳記佔據了
    raw_acc = df.iloc[:, 3:6].values.astype(float)
    
    # 計算量級 (Magnitude)
    mag = np.linalg.norm(raw_acc, axis=1)
    
    # 執行清理邏輯
    cleaned_mag = clean_signal_logic(mag.reshape(-1, 1), fs=FS).flatten()
    
    # 生成時間軸 (秒)
    time_secs = np.arange(len(mag)) / FS
    
    return time_secs, mag, cleaned_mag

# 繪圖設定
plt.figure(figsize=(15, 10))

for i, label in enumerate(TARGET_LABELS):
    data = get_data_and_time(label)
    if data is None:
        print(f"警告：找不到 {label} 的資料")
        continue
    
    time_axis, raw, cleaned = data
    
    # 因為濾波器是 0.1Hz 高通，會濾掉重力(DC)，
    # 我們將原始資料也減去平均值，以便在圖上對比雜訊
    raw_no_dc = raw - np.mean(raw)
    
    # 只顯示前 15 秒，讓波形更清楚
    mask = time_axis <= 15 
    t_plot = time_axis[mask]
    r_plot = raw_no_dc[mask]
    c_plot = cleaned[mask]

    plt.subplot(2, 1, i+1)
    
    # 1. 原始訊號 (去直流後)
    plt.plot(t_plot, r_plot, color='lightgray', alpha=0.7, label='Raw Signal (Centered)', linewidth=1)
    
    # 2. 清理後的訊號
    color = '#1f77b4' if 'Slow' in label else '#ff7f0e'
    plt.plot(t_plot, c_plot, color=color, label=f'Cleaned {label}', linewidth=2)
    
    # 3. 標示濾掉的雜訊 (Raw 與 Cleaned 的差異區塊)
    plt.fill_between(t_plot, c_plot, r_plot, color='red', alpha=0.3, label='Filtered Noise')
    
    plt.title(f'Waveform Analysis: {label} (20Hz Sampling)', fontsize=14)
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Acceleration (Magnitude)', fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(loc='upper right')
    
    # 設置 X 軸刻度間隔為 1 秒
    plt.xticks(np.arange(0, 16, 1))

plt.tight_layout()
plt.show()