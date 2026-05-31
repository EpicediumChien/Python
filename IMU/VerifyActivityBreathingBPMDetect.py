import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import rfft, rfftfreq

# ==========================================
# 1. 配置與驗證路徑
# ==========================================
VERIFY_FILES = {
    "Fast Walk": r"C:\Git\Python\IMU\data\Movement Verify Samples\Fast Walk\imu_YAHBOOM_20260322_172920.csv",
    "Slow Walk": r"C:\Git\Python\IMU\data\Movement Verify Samples\Slow Walk\imu_YAHBOOM_20260322_172141.csv",
    "Static Stand": r"C:\Git\Python\IMU\data\Movement Verify Samples\Static Stand\imu_YAHBOOM_20260322_173735.csv"
}

# 假設當時實驗設定的目標呼吸頻率 (請根據你當初錄製時的節拍器設定修改)
# 如果你忘記了，通常靜止/慢走測試會設定在 15 或 18 BPM
GROUND_TRUTH_BPM = 18.0  

TARGET_FREQ = 20
WINDOW_SIZE = 200  # 驗證時用較大的視窗 (30秒) 增加 FFT 頻率解析度

# ==========================================
# 2. 沿用你的核心預處理邏輯 (確保一致性)
# ==========================================
def apply_agc_logic(sig, fs=20, window_sec=10):
    window_size = int(fs * window_sec)
    rolling_std = pd.Series(sig).rolling(window=window_size, center=True).std()
    rolling_std = rolling_std.bfill().ffill().replace(0, 1e-6)
    return sig / rolling_std.values

def clean_signal_logic(data, fs=20):
    cleaned_data = np.zeros_like(data)
    nyq = 0.5 * fs
    b, a = signal.butter(2, [0.08/nyq, 0.85/nyq], btype='band')
    savgol_win = 15
    for i in range(data.shape[1]):
        feat_filt = signal.filtfilt(b, a, data[:, i])
        feat_agc = apply_agc_logic(feat_filt, fs=fs, window_sec=10)
        cleaned_data[:, i] = signal.savgol_filter(feat_agc, window_length=savgol_win, polyorder=2)
    return cleaned_data

# ==========================================
# 3. BPM 提取邏輯 (頻域分析)
# ==========================================
def estimate_bpm_from_file(file_path):
    df = pd.read_csv(file_path)
    df.columns = [c.lower().strip() for c in df.columns]
    
    # 重採樣至 20Hz
    ts_col = next(c for c in ['unix_timestamp', 'timestamp', 'unixtimestamp'] if c in df.columns)
    df['dt'] = pd.to_datetime(df[ts_col], unit='ms')
    df_res = df.set_index('dt').resample('50ms').mean().interpolate().ffill().bfill()
    
    # 提取特徵 (使用與訓練一致的 AccZ，通常對呼吸最敏感)
    acc_mag = np.linalg.norm(df_res[['accx', 'accy', 'accz']].values, axis=1, keepdims=True)
    raw_data = df_res[['accx', 'accy', 'accz', 'gyrox', 'gyroy', 'gyroz', 'roll', 'pitch', 'yaw']].values
    combined = np.hstack([raw_data, acc_mag, np.zeros_like(acc_mag)]) # 補齊 11 軸
    
    # 清洗訊號
    processed = clean_signal_logic(combined, fs=TARGET_FREQ)
    
    # 取中間段 30 秒數據進行 FFT (避開開頭不穩定的數據)
    mid = len(processed) // 2
    window = processed[mid - WINDOW_SIZE//2 : mid + WINDOW_SIZE//2, 2] # 取 AccZ 軸
    
    # FFT
    n = len(window)
    yf = np.abs(rfft(window))
    xf = rfftfreq(n, 1/TARGET_FREQ)
    
    # 鎖定呼吸頻率區間 [0.1Hz ~ 0.75Hz] (即 6 ~ 45 BPM)
    idx = np.where((xf >= 0.1) & (xf <= 0.75))[0]
    peak_freq = xf[idx][np.argmax(yf[idx])]
    est_bpm = peak_freq * 60
    
    return est_bpm, window, xf[idx], yf[idx]

# ==========================================
# 4. 執行驗證與計算誤差
# ==========================================
print(f"{'Activity':<15} | {'Ground Truth':<12} | {'Estimated BPM':<15} | {'Error (%)'}")
print("-" * 65)

results = []
fig, axs = plt.subplots(3, 2, figsize=(15, 12))

for i, (mode, path) in enumerate(VERIFY_FILES.items()):
    est_bpm, wave, xf_plot, yf_plot = estimate_bpm_from_file(path)
    error = abs(est_bpm - GROUND_TRUTH_BPM)
    error_pct = (error / GROUND_TRUTH_BPM) * 100
    
    print(f"{mode:<15} | {GROUND_TRUTH_BPM:<12.1f} | {est_bpm:<15.2f} | {error_pct:.2f}%")
    
    # 繪製時域訊號 (顯示呼吸波形是否清晰)
    axs[i, 0].plot(wave, color='blue', alpha=0.7)
    axs[i, 0].set_title(f"{mode} - Cleaned Respiration Wave")
    axs[i, 0].set_ylabel("Normalized Amp")
    
    # 繪製頻譜圖 (顯示是否存在明顯的主峰)
    axs[i, 1].plot(xf_plot * 60, yf_plot, color='red')
    axs[i, 1].axvline(GROUND_TRUTH_BPM, color='green', linestyle='--', label='Target')
    axs[i, 1].set_title(f"{mode} - Frequency Spectrum (Peak: {est_bpm:.2f} BPM)")
    axs[i, 1].set_xlabel("BPM")
    axs[i, 1].legend()

plt.tight_layout()
plt.show()