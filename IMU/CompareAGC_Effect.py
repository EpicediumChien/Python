import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import signal

# --- 1. 核心邏輯 ---

def apply_agc_logic(sig, fs=20, window_sec=10):
    window_size = int(fs * window_sec)
    rolling_std = pd.Series(sig).rolling(window=window_size, center=True).std()
    rolling_std = rolling_std.bfill().ffill().replace(0, 1e-6)
    return sig / rolling_std.values

def get_signal_stages(raw_sig, fs=20):
    """
    分解處理步驟以進行對比
    """
    nyq = 0.5 * fs
    # 1. 帶通濾波器 [0.08Hz - 0.85Hz]
    b, a = signal.butter(2, [0.08/nyq, 0.85/nyq], btype='band')
    
    # A. 濾波後 (Before AGC)
    filtered = signal.filtfilt(b, a, raw_sig)
    
    # B. AGC 處理後 (After AGC)
    after_agc = apply_agc_logic(filtered, fs=fs, window_sec=10)
    
    # C. SavGol 平滑 (最終輸出)
    try:
        final = signal.savgol_filter(after_agc, window_length=15, polyorder=2)
    except:
        final = after_agc
        
    return filtered, final

# --- 2. 比較繪圖邏輯 ---

def compare_agc_effect(folder_path, file_static, file_fast):
    fs = 20
    files = [file_static, file_fast]
    titles = ["STATIC STAND", "FAST WALK"]
    colors = ['green', 'red']
    
    # 建立 3(列) x 2(行) 的圖表
    fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex='col')

    for col, file_name in enumerate(files):
        file_path = os.path.join(folder_path, file_name)
        if not os.path.exists(file_path):
            print(f"File not found: {file_name}")
            continue
            
        df = pd.read_csv(file_path)
        df.columns = [c.lower().strip() for c in df.columns]
        acc_cols = [c for c in df.columns if 'acc' in c][:3]
        
        # 1. Raw Magnitude
        raw_values = df[acc_cols].values
        magnitude = np.sqrt(np.sum(raw_values**2, axis=1))
        time_axis = np.arange(len(magnitude)) / fs
        
        # 2. 取得不同階段的訊號
        before_agc, after_agc = get_signal_stages(magnitude, fs=fs)

        # --- 繪圖第一行：RAW ---
        axes[0, col].plot(time_axis, magnitude, color='gray', alpha=0.5)
        axes[0, col].set_title(f"RAW: {titles[col]}", fontsize=14, fontweight='bold')
        if col == 0: # Static Zoom
            m_mean = np.mean(magnitude)
            axes[0, col].set_ylim(m_mean - 0.05, m_mean + 0.05)
        
        # --- 繪圖第二行：Before AGC (Filtered only) ---
        axes[1, col].plot(time_axis, before_agc, color='blue', linewidth=1)
        axes[1, col].set_title(f"BEFORE AGC (Bandpass Filtered)", fontsize=12)
        if col == 0:
            axes[1, col].set_ylabel("Tiny Amplitude", color='blue')
        
        # --- 繪圖第三行：After AGC (Normalized) ---
        axes[2, col].plot(time_axis, after_agc, color=colors[col], linewidth=1.2)
        axes[2, col].set_title(f"AFTER AGC (Normalized Waveform)", fontsize=12)
        axes[2, col].set_xlabel("Time (seconds)")
        if col == 0:
            axes[2, col].set_ylabel("Standardized Amp", color=colors[col])

    plt.suptitle("AGC Impact Analysis: Static Stand vs. Fast Walk", fontsize=18, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# --- 執行 ---
target_folder = r"C:\Git\Python\IMU\data\Motion Artifact"
compare_agc_effect(target_folder, "Static Stand.csv", "Fast Walk.csv")