import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import signal
from scipy.fft import fft, fftfreq

# --- 1. 核心演算法邏輯 (保持不變) ---

def apply_agc_logic(sig, fs=20, window_sec=10):
    window_size = int(fs * window_sec)
    rolling_std = pd.Series(sig).rolling(window=window_size, center=True).std()
    rolling_std = rolling_std.bfill().ffill().replace(0, 1e-6)
    return sig / rolling_std.values

def clean_signal_logic(data_mag, fs=20):
    nyq = 0.5 * fs
    b, a = signal.butter(2, [0.08/nyq, 0.85/nyq], btype='band')
    feat_filt = signal.filtfilt(b, a, data_mag)
    feat_agc = apply_agc_logic(feat_filt, fs=fs, window_sec=10)
    try:
        final = signal.savgol_filter(feat_agc, window_length=15, polyorder=2)
    except:
        final = feat_agc
    return final, feat_filt

def get_fft_spectrum(sig, fs=20):
    """計算並回傳頻譜數據 (用於繪圖)"""
    n = len(sig)
    window = np.hanning(n)
    yf = np.abs(fft((sig - np.mean(sig)) * window))
    xf = fftfreq(n, 1/fs)
    # 只取正頻率區間
    mask = (xf >= 0.05) & (xf <= 1.5) 
    return xf[mask] * 60, yf[mask] # 回傳 BPM 和 能量值

def get_bpm_via_fft(sig, fs=20):
    xf_bpm, yf = get_fft_spectrum(sig, fs)
    if len(yf) == 0: return 0
    return xf_bpm[np.argmax(yf)]

def estimate_displacement_robust(acc_before_agc, bpm, fs=20):
    if bpm <= 5: return 0
    f_target = bpm / 60.0
    n = len(acc_before_agc)
    window = np.hanning(n)
    yf = np.fft.fft((acc_before_agc - np.mean(acc_before_agc)) * window)
    xf = np.fft.fftfreq(n, 1/fs)
    idx = np.argmin(np.abs(xf - f_target))
    acc_amp_g = (np.abs(yf[idx]) / n) * 4.0
    acc_amp_ms2 = acc_amp_g * 9.80665
    omega = 2 * np.pi * f_target
    disp_mm = (acc_amp_ms2 / (omega**2)) * 2 * 1000
    if disp_mm > 10.0: disp_mm = 8.0 + np.log10(disp_mm) 
    return disp_mm

# --- 2. 主分析與繪圖邏輯 (拆分為兩張圖) ---

def run_comprehensive_analysis(file_path):
    fs = 20
    df = pd.read_csv(file_path)
    df.columns = [c.lower().strip() for c in df.columns]
    
    acc_raw = np.sqrt(np.sum(df[[c for c in df.columns if 'acc' in c][:3]].values**2, axis=1))
    gyro_raw = np.sqrt(np.sum(df[[c for c in df.columns if 'gyro' in c][:3]].values**2, axis=1))
    
    acc_clean, acc_phys = clean_signal_logic(acc_raw, fs=fs)
    gyro_clean, _ = clean_signal_logic(gyro_raw, fs=fs)
    
    bpm_acc = get_bpm_via_fft(acc_clean, fs=fs)
    bpm_gyro = get_bpm_via_fft(gyro_clean, fs=fs)
    disp_mm = estimate_displacement_robust(acc_phys, bpm_acc, fs=fs)
    
    time_axis = np.arange(len(acc_raw)) / fs

    # --- 圖表 1: 時域分析 (Time Domain) ---
    fig1, axes1 = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    axes1[0].plot(time_axis, acc_raw, color='gray', alpha=0.5); axes1[0].set_title("Raw Accelerometer Magnitude (g)")
    axes1[1].plot(time_axis, acc_clean, color='dodgerblue', label=f'Acc BPM: {bpm_acc:.1f}'); axes1[1].legend(); axes1[1].set_title("Processed Acc Signal")
    axes1[2].plot(time_axis, gyro_raw, color='gray', alpha=0.5); axes1[2].set_title("Raw Gyroscope Magnitude (deg/s)")
    axes1[3].plot(time_axis, gyro_clean, color='crimson', label=f'Gyro BPM: {bpm_gyro:.1f}'); axes1[3].legend(); axes1[3].set_title("Processed Gyro Signal")
    fig1.suptitle(f"Time Domain Analysis: {os.path.basename(file_path)}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # --- 圖表 2: 頻域分析 (Frequency Domain / FFT) ---
    fig2, axes2 = plt.subplots(2, 1, figsize=(10, 8))
    
    # Acc FFT
    xf_acc, yf_acc = get_fft_spectrum(acc_clean, fs)
    axes2[0].fill_between(xf_acc, yf_acc, color='dodgerblue', alpha=0.3)
    axes2[0].plot(xf_acc, yf_acc, color='dodgerblue', linewidth=1.5)
    axes2[0].axvline(bpm_acc, color='orange', linestyle='--', label=f'Peak: {bpm_acc:.1f} BPM')
    axes2[0].set_title("Accelerometer Frequency Spectrum (FFT)", fontweight='bold')
    axes2[0].set_ylabel("Magnitude")
    axes2[0].legend()

    # Gyro FFT
    xf_gyro, yf_gyro = get_fft_spectrum(gyro_clean, fs)
    axes2[1].fill_between(xf_gyro, yf_gyro, color='crimson', alpha=0.3)
    axes2[1].plot(xf_gyro, yf_gyro, color='crimson', linewidth=1.5)
    axes2[1].axvline(bpm_gyro, color='black', linestyle='--', label=f'Peak: {bpm_gyro:.1f} BPM')
    axes2[1].set_title("Gyroscope Frequency Spectrum (FFT)", fontweight='bold')
    axes2[1].set_xlabel("Breathing Rate (BPM)")
    axes2[1].set_ylabel("Magnitude")
    axes2[1].legend()

    fig2.suptitle(f"Spectral Analysis (FFT): {os.path.basename(file_path)}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    plt.show()

if __name__ == "__main__":
    target_file = r"C:\Git\Python\IMU\data\YoungBreath\imu_log_20260124_195137.csv"
    run_comprehensive_analysis(target_file)