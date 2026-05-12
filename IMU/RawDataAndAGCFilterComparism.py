import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# --- 1. 你提供的原始邏輯 ---

def apply_agc_logic(sig, fs=20, window_sec=5):
    window_size = int(fs * window_sec)
    rolling_std = pd.Series(sig).rolling(window=window_size, center=True).std()
    # 填充邊界並避免除以零
    rolling_std = rolling_std.bfill().ffill().replace(0, 1e-6)
    return sig / rolling_std.values

def clean_signal_logic(data, fs=20):
    cleaned_data = np.zeros_like(data)
    nyq = 0.5 * fs
    
    # 1. 原始濾波器 [0.08Hz - 0.85Hz]
    b, a = signal.butter(2, [0.08/nyq, 0.85/nyq], btype='band')
    
    # 2. 原始平滑視窗 (Savgol)
    savgol_win = 15 
    
    for i in range(data.shape[1]):
        # A. 帶通濾波
        feat_filt = signal.filtfilt(b, a, data[:, i])
        
        # B. 原始 AGC (10秒視窗)
        feat_agc = apply_agc_logic(feat_filt, fs=fs, window_sec=10)
        
        # C. 原始 SavGol
        try:
            cleaned_data[:, i] = signal.savgol_filter(feat_agc, window_length=savgol_win, polyorder=2)
        except:
            cleaned_data[:, i] = feat_agc
            
    return cleaned_data

# --- 2. 檔案讀取與對比顯示 ---

def verify_straw_compare_sample(file_path):
    fs = 20
    
    # 讀取檔案
    df = pd.read_csv(file_path)
    # 欄位名稱標準化（轉小寫並去空白）
    df.columns = [c.lower().strip() for c in df.columns]
    
    # 抓取加速度欄位 (accx, accy, accz)
    acc_cols = [c for c in df.columns if 'acc' in c][:3]
    print(f"使用的加速度欄位: {acc_cols}")
    
    # 1. 計算原始合向量 (Raw Magnitude)
    raw_acc_values = df[acc_cols].values
    raw_magnitude = np.sqrt(np.sum(raw_acc_values**2, axis=1))
    
    # 2. 應用你的清理邏輯 (Cleaned Signal)
    # reshape(-1, 1) 是因為 clean_signal_logic 預期輸入為 2D array (samples, channels)
    cleaned_signal = clean_signal_logic(raw_magnitude.reshape(-1, 1), fs=fs).flatten()
    
    # 設定時間軸
    time_axis = np.arange(len(raw_magnitude)) / fs
    
    # --- 3. 繪圖對比 ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # 左側圖表：Raw Data (Zoomed for breathing)
    axes[0].plot(time_axis, raw_magnitude, color='gray', linewidth=1)
    mag_mean = np.mean(raw_magnitude)
    axes[0].set_ylim(mag_mean - 0.05, mag_mean + 0.05) # 縮小範圍顯示微波
    axes[0].set_title(f"LEFT: Raw Data (Magnitude)\n[Zoomed at {mag_mean:.2f}g]", fontsize=14, color='green')
    axes[0].set_ylabel("Amplitude (g)")
    axes[0].set_xlabel("Time (seconds)")
    axes[0].grid(True, alpha=0.3)
    axes[0].annotate('Breathing Micro-waves', xy=(time_axis[len(time_axis)//2], mag_mean), 
                     xytext=(time_axis[len(time_axis)//2]+2, mag_mean+0.03),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1))

    # 右側圖表：Processed Data
    axes[1].plot(time_axis, cleaned_signal, color='blue', linewidth=1.5)
    axes[1].set_title("RIGHT: Processed Data\n[Bandpass + AGC + SavGol]", fontsize=14, color='blue')
    axes[1].set_ylabel("Normalized Amplitude")
    axes[1].set_xlabel("Time (seconds)")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f"Algorithm Verification: {file_path.split('/')[-1]}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# --- 執行驗證 ---
file_to_check = r"C:\Git\Python\IMU\data\StrawCompare\StaticSit_imu_YAHBOOM_20260320_002717.csv"
verify_straw_compare_sample(file_to_check)