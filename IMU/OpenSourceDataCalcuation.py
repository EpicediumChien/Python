import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# ==========================================
# 1. Configuration
# ==========================================
DATA_DIR = os.path.join('data', 'Open Source Chest Data')
FILE_PATTERN = "Chest*.csv"
FS = 25  # Updated to 25Hz

def apply_agc(sig, window_size=FS*4):
    rolling_std = pd.Series(sig).rolling(window=int(window_size), center=True).std()
    rolling_std = rolling_std.bfill().ffill().replace(0, 1e-6)
    return sig / rolling_std

def process_yahboom_file(file_path):
    # Load data - Added sep=';' to handle your specific file format
    df = pd.read_csv(file_path, sep=';')
    
    # --- TIME CALCULATION ---
    # Based on: "time index is from 1 and the file is 25 Hz"
    # Time (seconds) = (index - 1) / 25
    df['Time_Sec'] = (df['time index'] - 1) / FS
    duration_sec = len(df) / FS
    
    # --- SIGNAL EXTRACTION ---
    # Using the specific lowercase column names from your header
    try:
        raw_sig = np.linalg.norm(df[['acc_x', 'acc_y', 'acc_z']].values, axis=1)
    except KeyError:
        print(f"Error: Could not find acc_x/y/z in {file_path}. Check column names.")
        return

    # ==========================================
    # 2. Adjusted Processing for 25Hz
    # ==========================================
    nyq = 0.5 * FS
    
    # FILTER: [0.08Hz (4.8 BPM) to 0.85Hz (51 BPM)]
    b, a = signal.butter(2, [0.08/nyq, 0.85/nyq], btype='band')
    sig_filt = signal.filtfilt(b, a, raw_sig)

    # AGC: 10-second window (now 250 samples)
    sig_agc = apply_agc(sig_filt, window_size=FS * 10) 

    # SMOOTHING: Increased window for 25Hz
    # 27 samples at 25Hz is approx 1.08s (must be an odd number)
    sig_smooth = signal.savgol_filter(sig_agc, window_length=27, polyorder=2)

    # PEAK DETECTION:
    # distance=20 (0.8s): Safe for up to 60-70 BPM.
    # prominence=1.5: Keeps detection strict against noise.
    peaks, _ = signal.find_peaks(sig_smooth, distance=20, prominence=1.5)
    
    # --- BPM Calculation ---
    if duration_sec > 5:
        bpm = (len(peaks) / duration_sec) * 60
    else:
        bpm = 0

    # ==========================================
    # 4. Status Logic
    # ==========================================
    # if 12 <= bpm <= 20:
    #     status_text = "NORMAL"
    #     status_color = "green"
    # else:
    #     status_text = "ABNORMAL"
    #     status_color = "red"
        # --- Status Logic ---
    # We define 'status' here so the print statement can find it
    if 12 <= bpm <= 20:
        status = "NORMAL"
    else:
        status = "ABNORMAL"


    # # ==========================================
    # # 6. Visualization
    # # ==========================================
    # plt.figure(figsize=(16, 7))
    
    # # X-axis is now Time in Seconds instead of Clock Time
    # plt.plot(df['Time_Sec'], sig_smooth, label='Processed Respiration', color='#2980b9', lw=2)
    # plt.scatter(df['Time_Sec'].iloc[peaks], sig_smooth[peaks], color='red', s=80, 
    #             label=f'Breaths Detected: {len(peaks)}', zorder=5)

    # plt.title(f"FILE: {os.path.basename(file_path)}\n"
    #           f"Rate: {bpm:.2f} BPM | STATUS: {status_text}", 
    #           fontsize=14, fontweight='bold', color=status_color)
    
    # plt.xlabel("Time (Seconds)")
    # plt.ylabel("Normalized Amplitude")
    # plt.grid(True, alpha=0.3, linestyle='--')
    # plt.legend()
    
    # plt.tight_layout()
    # plt.show()

    # ==========================================
    # 3. Output Result
    # ==========================================
    filename = os.path.basename(file_path)
    print(f"{status:<10} | {filename:<30} | {bpm:>6.2f} BPM | Breaths: {len(peaks):>3}")

if __name__ == "__main__":
    # Get all files matching the pattern
    files = glob.glob(os.path.join(DATA_DIR, FILE_PATTERN))
    
    if not files:
        print(f"No files found in {DATA_DIR} matching {FILE_PATTERN}")
        print("Check if your DATA_DIR path and FILE_PATTERN are correct.")
    else:
        # Print a header for the output table
        print("-" * 75)
        print(f"{'STATUS':<10} | {'FILENAME':<30} | {'RATE':<10} | {'COUNT'}")
        print("-" * 75)
        
        for f in sorted(files):
            process_yahboom_file(f)
        
        print("-" * 75)