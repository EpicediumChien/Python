import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
import os

# --- PATHS ---
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'TrainingData', 'Static Sit', 'Normal Breath Example.csv')

# --- CONFIGURATION ---
TARGET_AXIS = 'accZ'   # Z-axis had the best signal in your previous plot
LOW_CUT = 0.1          # Lower limit for breathing
HIGH_CUT = 0.8         # Upper limit (removes heartbeat noise)

# --- PROCESS ---
if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    
    # 1. Calculate Sampling Rate
    avg_diff_ms = df['timestamp'].diff().mean()
    fs = 1000.0 / avg_diff_ms if avg_diff_ms > 0 else 50.0
    print(f"Sampling Rate: {fs:.1f} Hz")
    print(f"Total Duration: {len(df)/fs:.1f} seconds")
    
    # 2. Extract Z-Axis
    raw_signal = df[TARGET_AXIS].values
    
    # 3. Smooth the "Staircase" steps (Moving Average)
    # 0.5s window to melt the jagged digital steps
    window_size = int(fs * 0.5) 
    smoothed_series = pd.Series(raw_signal).rolling(window=window_size, center=True).mean().fillna(method='bfill').fillna(method='ffill')
    smoothed_data = smoothed_series.values

    # 4. Bandpass Filter
    def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        y = filtfilt(b, a, data)
        return y
        
    final_signal = butter_bandpass_filter(smoothed_data, LOW_CUT, HIGH_CUT, fs, order=4)
    
    # 5. Normalize
    normalized = (final_signal - np.mean(final_signal)) / np.std(final_signal)
    
    # 6. Count Breaths (Find Peaks)
    # distance=fs*2 means peaks must be at least 2 seconds apart (max 30 BPM)
    peaks, _ = find_peaks(normalized, height=0.5, distance=fs*2.0)
    num_breaths = len(peaks)
    duration_min = (len(normalized)/fs) / 60.0
    bpm = num_breaths / duration_min
    
    # --- PLOT THE WHOLE DURATION ---
    t = np.linspace(0, len(normalized)/fs, len(normalized))
    
    plt.figure(figsize=(12, 6))
    plt.plot(t, normalized, linewidth=2, color='#1f77b4', label='Processed Breath Signal')
    
    # Plot Red Dots on Detected Peaks
    plt.plot(t[peaks], normalized[peaks], "x", color='red', markersize=10, label='Detected Breaths')
    
    plt.title(f"Full 1-Minute Deep Breathing Analysis\nCounted {num_breaths} Breaths (~{bpm:.1f} BPM)", fontsize=14, fontweight='bold')
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Normalized Amplitude", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

else:
    print("File not found.")