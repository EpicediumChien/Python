import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import signal

# ==========================================
# 1. Configuration
# ==========================================
DATA_DIR = os.path.join('data', 'StrawCompare')
# DATA_DIR = os.path.join('data', 'MAE')
FILE_PATTERN = "*.csv"
FS = 20  

def apply_agc(sig, window_size=FS*4):
    rolling_std = pd.Series(sig).rolling(window=int(window_size), center=True).std()
    rolling_std = rolling_std.bfill().ffill().replace(0, 1e-6)
    return sig / rolling_std

def process_yahboom_file(file_path):
    # Load data
    df = pd.read_csv(file_path)
    
    # --- CRITICAL FIX: Time Calculation ---
    # Use UnixTimestamp (ms) for duration calculation to avoid string parsing errors
    # Based on your sample, UnixTimestamp is in milliseconds
    start_time_ms = df['UnixTimestamp'].iloc[0]
    end_time_ms = df['UnixTimestamp'].iloc[-1]
    duration_sec = (end_time_ms - start_time_ms) / 1000.0
    
    # 1. Convert Unix Timestamp (ms) to UTC
    # 2. Add 8 hours (+8) to match Taiwan/Local time
    df['DateTime'] = pd.to_datetime(df['UnixTimestamp'], unit='ms') + pd.to_timedelta(8, unit='h')
    
    # --- FIX 1: Correct Column Alignment ---
    # Try to get AccX, AccY, AccZ by name first.
    # If KeyError (e.g., due to column shift in the CSV as noted previously with duplicate UnixTimestamp),
    # fall back to iloc based on the observed shifted structure in your sample data.
    try:
        raw_sig = np.linalg.norm(df[['AccX', 'AccY', 'AccZ']].values, axis=1)
    except KeyError:
        print(f"Warning: Columns 'AccX', 'AccY', 'AccZ' not found directly. Attempting iloc for {file_path}")
        # Assuming the structure is: DateTime, UnixTimestamp, UnixTimestamp (dup), AccX, AccY, AccZ
        # So AccX would be at index 3 (0-indexed).
        raw_sig = np.linalg.norm(df.iloc[:, [3, 4, 5]].values, axis=1)

    # ==========================================
    # 2. Universal Balanced Processing (10 - 50 BPM)
    # ==========================================
    nyq = 0.5 * FS
    
    # FILTER: [0.08Hz (4.8 BPM) to 0.85Hz (51 BPM)]
    # This covers your entire range safely.
    b, a = signal.butter(2, [0.08/nyq, 0.85/nyq], btype='band')
    sig_filt = signal.filtfilt(b, a, raw_sig)

    # AGC: 10-second window to stabilize the slow 10 BPM signal.
    sig_agc = apply_agc(sig_filt, window_size=FS * 10) 

    # SMOOTHING: 11 samples (1.1s)
    # This is the "Goldilocks" value. It's smooth enough to hide 
    # the wiggles in 10 BPM, but sharp enough for 46 BPM.
    sig_smooth = signal.savgol_filter(sig_agc, window_length=11, polyorder=2)

    # PEAK DETECTION:
    # distance=10 (1.0s): Fast enough to catch 46 BPM (1.3s period).
    # prominence=1.5: THE CRITICAL FIX. 
    # This ignores all those small red dots you see in your "Worse" image.
    peaks, _ = signal.find_peaks(sig_smooth, distance=10, prominence=1.5)
    
    # --- BPM Calculation ---
    if duration_sec > 5:
        bpm = (len(peaks) / duration_sec) * 60
    else:
        bpm = 0

    # ==========================================
    # 4. Status Logic (12-20 BPM)
    # ==========================================
    if 12 <= bpm <= 20:
        status_text = "NORMAL"
        status_color = "green"
    else:
        status_text = "ABNORMAL"
        status_color = "red"

    # ==========================================
    # 6. Visualization
    # ==========================================
    plt.figure(figsize=(16, 7))
    
    plt.plot(df['DateTime'], sig_smooth, label='Processed Respiration (AGC)', color='#2980b9', lw=2)
    plt.scatter(df['DateTime'].iloc[peaks], sig_smooth[peaks], color='red', s=80, 
                label=f'Breaths Detected: {len(peaks)}', zorder=5)

    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=30)
    
    plt.title(f"YAHBOOM IMU ANALYSIS: {os.path.basename(file_path)}\n"
              f"Rate: {bpm:.2f} BPM | STATUS: {status_text}", 
              fontsize=14, fontweight='bold', color=status_color)
    
    plt.xlabel("Clock Time (HH:MM:SS)")
    plt.ylabel("Normalized Amplitude")
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    files = glob.glob(os.path.join(DATA_DIR, FILE_PATTERN))
    for f in sorted(files):
        process_yahboom_file(f)