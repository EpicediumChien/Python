import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import signal
from scipy import stats  # 新增：用於統計檢定

# --- Constants ---
NORMAL_FILE = os.path.join('data', 'StrawCompare', '30 BPM.csv')
ABNORMAL_FILE = os.path.join('data', 'StrawCompare', '30 BPM - Straw.csv')

FS = 20 

# --- New Logic Functions ---

def apply_agc_logic(sig, fs=20, window_sec=5):
    window_size = int(fs * window_sec)
    rolling_std = pd.Series(sig).rolling(window=window_size, center=True).std()
    rolling_std = rolling_std.bfill().ffill().replace(0, 1e-6)
    return sig / rolling_std.values

def clean_signal_logic(data, fs=20):
    # 修正：確保 data 是一維陣列，如果是多維則處理方式不同
    if len(data.shape) > 1:
        input_is_2d = True
        cols = data.shape[1]
        cleaned_data = np.zeros_like(data)
    else:
        input_is_2d = False
        cols = 1
        cleaned_data = np.zeros_like(data)

    nyq = 0.5 * fs
    b, a = signal.butter(2, [0.08/nyq, 0.85/nyq], btype='band')
    savgol_win = 15 
    
    # 修正：根據維度處理
    def process_vector(vec):
        feat_filt = signal.filtfilt(b, a, vec)
        feat_agc = apply_agc_logic(feat_filt, fs=fs, window_sec=10)
        try:
            return signal.savgol_filter(feat_agc, window_length=savgol_win, polyorder=2)
        except:
            return feat_agc

    if input_is_2d:
        for i in range(cols):
            cleaned_data[:, i] = process_vector(data[:, i])
    else:
        cleaned_data = process_vector(data)
            
    return cleaned_data

def get_sliding_variance(sig, fs, window_sec=10, step_sec=0.5):
    win_size = int(fs * window_sec)
    step_size = int(fs * step_sec)
    variances = []
    for i in range(0, len(sig) - win_size, step_size):
        segment = sig[i : i + win_size]
        variances.append(np.var(segment))
    return variances

# --- Main Processing ---

def process_data(file_path):
    df = pd.read_csv(file_path)
    
    start_time_ms = df['UnixTimestamp'].iloc[0]
    end_time_ms = df['UnixTimestamp'].iloc[-1]
    duration_sec = (end_time_ms - start_time_ms) / 1000.0
    
    df['DateTime'] = pd.to_datetime(df['UnixTimestamp'], unit='ms') + pd.to_timedelta(8, unit='h')
    
    try:
        raw_mag = np.linalg.norm(df[['AccX', 'AccY', 'AccZ']].values, axis=1)
    except KeyError:
        raw_mag = np.linalg.norm(df.iloc[:, [3, 4, 5]].values, axis=1)

    sig_final = clean_signal_logic(raw_mag, fs=FS)

    # 計算方差樣本 (每 10 秒一個樣本，用於統計)
    var_samples = get_sliding_variance(sig_final, FS, window_sec=10, step_sec=0.5)
    avg_variance = np.var(sig_final)

    peaks, _ = signal.find_peaks(sig_final, distance=int(FS * 0.8), prominence=0.8)
    bpm = (len(peaks) / duration_sec) * 60
    
    status_text = "NORMAL" if 12 <= bpm <= 20 else "ABNORMAL"
    status_color = "green" if status_text == "NORMAL" else "red"
    
    return df['DateTime'], sig_final, peaks, bpm, status_text, status_color, avg_variance, var_samples

def compare_results(file1, file2):
    # 獲取兩組數據
    t1, s1, p1, b1, st1, c1, v1, samples1 = process_data(file1)
    t2, s2, p2, b2, st2, c2, v2, samples2 = process_data(file2)

    # --- 執行統計檢定 (Welch's t-test) ---
    t_stat, p_val = stats.ttest_ind(samples1, samples2, equal_var=False)
    
    print("-" * 30)
    print(f"統計驗證結果:")
    print(f"一般呼吸 (30 BPM) 方差: {v1:.6f}")
    print(f"受阻呼吸 (Straw) 方差: {v2:.6f}")
    print(f"變異增幅: {((v2-v1)/v1)*100:.2f}%")
    print(f"T-statistic: {t_stat:.4f}, P-value: {p_val:.6f}")
    if p_val < 0.05:
        print("結論: 顯著差異 (p < 0.05)")
    else:
        print("結論: 無顯著差異")
    print("-" * 30)

    # --- 繪圖 ---
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharey=True)
    
    data_list = [(t1, s1, p1, b1, st1, c1, v1, "Baseline: Normal"), 
                 (t2, s2, p2, b2, st2, c2, v2, "Stress Test: Straw")]

    for i, (times, sig, peaks, bpm, status, color, var, title) in enumerate(data_list):
        ax = axes[i]
        ax.plot(times, sig, label=f'Cleaned (Var: {var:.4f})', color='#2980b9', lw=1.5)
        ax.scatter(times.iloc[peaks], sig[peaks], color='red', s=60, label=f'Breaths: {len(peaks)}', zorder=5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax.set_title(f"{title}\nRate: {bpm:.2f} BPM | Variance: {var:.6f}", fontsize=13, fontweight='bold', color=color)
        ax.set_ylabel("Normalized Amplitude")
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right')

    plt.suptitle(f"Dyspnea Biomechanical Validation (p-value: {p_val:.6f})", fontsize=16)
    plt.xlabel("Clock Time (HH:MM:SS)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    if os.path.exists(NORMAL_FILE) and os.path.exists(ABNORMAL_FILE):
        compare_results(NORMAL_FILE, ABNORMAL_FILE)
    else:
        print("Check file paths: data/StrawCompare/...")