import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import signal

# --- File Paths ---
NORMAL_FILE = os.path.join('data', 'StrawCompare', 'StaticSit_imu_YAHBOOM_20260301_194414.csv')
ABNORMAL_FILE = os.path.join('data', 'StrawCompare', 'Straw_Dyspnea_YAHBOOM_20260311_233245.csv')

FS = 10

def apply_agc(sig, window_size=FS*10):
    """As per your requirement: 10-second window to stabilize slow signals"""
    rolling_std = pd.Series(sig).rolling(window=int(window_size), center=True).std()
    rolling_std = rolling_std.bfill().ffill().replace(0, 1e-6)
    return sig / rolling_std

def process_data(file_path):
    """Processes the file and returns all necessary components for plotting."""
    df = pd.read_csv(file_path)
    
    # Time Calculation
    start_time_ms = df['UnixTimestamp'].iloc[0]
    end_time_ms = df['UnixTimestamp'].iloc[-1]
    duration_sec = (end_time_ms - start_time_ms) / 1000.0
    
    # Time Conversion (UTC+8)
    df['DateTime'] = pd.to_datetime(df['UnixTimestamp'], unit='ms') + pd.to_timedelta(8, unit='h')
    
    # Column Alignment Fix
    try:
        raw_sig = np.linalg.norm(df[['AccX', 'AccY', 'AccZ']].values, axis=1)
    except KeyError:
        raw_sig = np.linalg.norm(df.iloc[:, [3, 4, 5]].values, axis=1)

    # --- REFINED PROCESSING PARAMETERS ---
    # Universal Balanced Processing
    nyq = 0.5 * FS
    
    # 1. Wider Filter: 0.07Hz (4 BPM) to 1.0Hz (60 BPM)
    b, a = signal.butter(2, [0.07/nyq, 1.0/nyq], btype='band')
    sig_filt = signal.filtfilt(b, a, raw_sig)

    # 2. AGC: Keep the 10s window to stabilize amplitude
    sig_agc = apply_agc(sig_filt, window_size=FS * 10) 

    # 3. LIGHTER SMOOTHING: Window 7 instead of 11 
    # (This stops merging fast breaths together)
    sig_smooth = signal.savgol_filter(sig_agc, window_length=7, polyorder=2)

    # 4. SENSITIVE PEAK DETECTION: 
    # prominence=0.8 (Catch smaller gasps)
    # distance=7 (Allow breaths as fast as 0.7s apart)
    peaks, _ = signal.find_peaks(sig_smooth, distance=7, prominence=0.8)
    
    bpm = (len(peaks) / duration_sec) * 60
    
    # Status: 12-20 is healthy resting. >20 is Tachypnea (Abnormal)
    status_text = "NORMAL" if 12 <= bpm <= 20 else "ABNORMAL"
    status_color = "green" if status_text == "NORMAL" else "red"
    
    return df['DateTime'], sig_smooth, peaks, bpm, status_text, status_color

def compare_results(file1, file2):
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharey=True)
    files = [file1, file2]
    titles = ["Baseline: Normal Sitting", "Stress Test: Straw Dyspnea"]

    for i, file_path in enumerate(files):
        times, sig, peaks, bpm, status, color = process_data(file_path)
        ax = axes[i]
        
        # Plot Waveform
        ax.plot(times, sig, label='Processed Respiration (AGC)', color='#2980b9', lw=1.5)
        
        # Plot Peaks
        ax.scatter(times.iloc[peaks], sig[peaks], color='red', s=60, 
                    label=f'Breaths: {len(peaks)}', zorder=5)

        # Formatting
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        
        ax.set_title(f"{titles[i]}\nRate: {bpm:.2f} BPM | STATUS: {status}", 
                     fontsize=13, fontweight='bold', color=color)
        ax.set_ylabel("Normalized Amplitude")
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right')

    plt.xlabel("Clock Time (HH:MM:SS)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Ensure files exist before running
    if os.path.exists(NORMAL_FILE) and os.path.exists(ABNORMAL_FILE):
        compare_results(NORMAL_FILE, ABNORMAL_FILE)
    else:
        print("One or both files not found. Please check paths.")