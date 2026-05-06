import pandas as pd
import numpy as np
from scipy import signal
import os
import glob
import re

# ==========================================
# 1. Aligned Configuration
# ==========================================
TARGET_FREQ = 25        
WINDOW_SIZE = 250       
STEP_SIZE = 25          
# These match your CSV exactly (lowercase/stripped)
FEATURES = ['accx', 'accy', 'accz', 'gyrox', 'gyroy', 'gyroz', 'roll', 'pitch', 'yaw']

# IMPORTANT: Point this to the folder containing the FIXED files!
DATA_DIR = r"C:/Git/Python/IMU/data/Time Example"

# ==========================================
# 2. Logic Functions
# ==========================================

def apply_agc_logic(sig, fs=25, window_sec=10):
    window_size = int(fs * window_sec)
    rolling_std = pd.Series(sig).rolling(window=window_size, center=True).std()
    rolling_std = rolling_std.bfill().ffill().replace(0, 1e-6)
    return sig / rolling_std.values

def clean_signal_logic(data, fs=25):
    cleaned_data = np.zeros_like(data)
    nyq = 0.5 * fs
    b, a = signal.butter(2, [0.08/nyq, 0.85/nyq], btype='band')
    savgol_win = 27 
    for i in range(data.shape[1]):
        feat_filt = signal.filtfilt(b, a, data[:, i])
        feat_agc = apply_agc_logic(feat_filt, fs=fs, window_sec=10)
        try:
            cleaned_data[:, i] = signal.savgol_filter(feat_agc, window_length=savgol_win, polyorder=2)
        except:
            cleaned_data[:, i] = feat_agc
    return cleaned_data

def get_dominant_bpm(window_data, fs=25):
    # Acc Magnitude is the last column we add (index 9)
    sig = window_data[:, 9] 
    sig = sig - np.mean(sig) 
    freqs = np.fft.rfftfreq(len(sig), d=1/fs)
    fft_mag = np.abs(np.fft.rfft(sig))
    idx = np.where((freqs >= 0.08) & (freqs <= 0.85))[0]
    if len(idx) == 0: return 0.0
    return freqs[idx][np.argmax(fft_mag[idx])] * 60

# ==========================================
# 3. Main Verification Loop
# ==========================================

if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        print(f"Error: Folder not found: {DATA_DIR}")
        exit()

    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if not files:
        print(f"No CSV files found in {DATA_DIR}")
        exit()

    results = []
    print(f"\n{'Target BPM':<12} | {'Detected BPM':<15} | {'Error (%)':<10}")
    print("-" * 45)

    for f in sorted(files, key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group())):
        filename = os.path.basename(f)
        target_bpm = int(re.search(r'\d+', filename).group())

        try:
            # 1. Load Data
            df = pd.read_csv(f)
            df.columns = [c.lower().strip() for c in df.columns]

            # 2. Find the correct timestamp column
            # Your header is 'UnixTimestamp', which becomes 'unixtimestamp'
            if 'unixtimestamp' in df.columns:
                ts_col = 'unixtimestamp'
            elif 'timestamp' in df.columns:
                ts_col = 'timestamp'
            else:
                # If neither is found, pick the 2nd column as fallback
                ts_col = df.columns[1]

            # 3. Resample to 25Hz (40ms)
            df['dt'] = pd.to_datetime(df[ts_col], unit='ms')
            df = df.set_index('dt')
            
            # Select only the features we need
            df_sensors = df.reindex(columns=FEATURES)
            df_res = df_sensors.resample('40ms').mean().interpolate().ffill().bfill()

            # 4. Extract Acc Magnitude
            acc_cols = [c for c in df_res.columns if 'acc' in c][:3]
            acc_mag = np.linalg.norm(df_res[acc_cols].values, axis=1, keepdims=True)
            
            # Combine 9 Features + 1 Acc Magnitude = 10 Columns
            raw_data = df_res.values.astype(np.float32)
            combined = np.hstack([raw_data, acc_mag]) 
            
            # 5. Clean
            processed_data = clean_signal_logic(combined, fs=TARGET_FREQ)

            # 6. Analyze windows
            window_bpms = []
            for i in range(0, len(processed_data) - WINDOW_SIZE, STEP_SIZE):
                window = processed_data[i : i + WINDOW_SIZE]
                bpm = get_dominant_bpm(window, fs=TARGET_FREQ)
                window_bpms.append(bpm)
            
            if not window_bpms:
                print(f"{target_bpm:<12} | File too short  | N/A")
                continue

            detected_bpm = np.median(window_bpms)
            error = abs(detected_bpm - target_bpm) / target_bpm * 100

            print(f"{target_bpm:<12} | {detected_bpm:<15.2f} | {error:<10.2f}%")
            results.append({'target': target_bpm, 'detected': detected_bpm})

        except Exception as e:
            print(f"Failed {filename}: {e}")

    if results:
        avg_error = np.mean([abs(r['detected']-r['target'])/r['target']*100 for r in results])
        print("-" * 45)
        print(f"Overall Pre-processing Accuracy: {100 - avg_error:.2f}%")