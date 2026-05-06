import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks

# --- File Paths ---
NORMAL_FILE = os.path.join('data', 'StrawCompare', 'StaticSit_imu_YAHBOOM_20260320_002717.csv')
ABNORMAL_FILE = os.path.join('data', 'StrawCompare', 'Straw_Dyspnea_imu_YAHBOOM_20260320_003148.csv')

FS = 10  # Sampling Frequency

def get_respiratory_data(file_path):
    # Load data (assuming columns 3,4,5 are Accel X,Y,Z)
    df = pd.read_csv(file_path)
    raw_sig = np.linalg.norm(df.iloc[:, [3, 4, 5]].values, axis=1) 
    
    # 60-second Window
    window_size = FS * 60
    if len(raw_sig) > window_size:
        start = len(raw_sig) // 2 - (window_size // 2)
        sig_crop = raw_sig[int(start):int(start + window_size)]
    else:
        sig_crop = raw_sig
        
    # 1. Detrend and Filter (Use a higher order for sharper cutoff)
    sig_detrend = signal.detrend(sig_crop)
    nyq = 0.5 * FS
    # Bandpass [0.1Hz (6 BPM) to 0.8Hz (48 BPM)]
    b, a = signal.butter(4, [0.1/nyq, 0.8/nyq], btype='band')
    sig_filt = signal.filtfilt(b, a, sig_detrend)
    
    # 2. Simple Normalization (Instead of rolling AGC which can distort peaks)
    sig_norm = (sig_filt - np.mean(sig_filt)) / np.std(sig_filt)
    
    # 3. Time-Domain Peak Detection (More accurate for Dyspnea)
    # distance=FS*1.2 ensures we don't count wiggles as breaths (min 1.2s between breaths)
    peaks, _ = find_peaks(sig_norm, distance=FS*1.2, prominence=0.5)
    bpm_peaks = (len(peaks) / (len(sig_norm)/FS)) * 60
    
    # 4. FFT Calculation for Visualization
    N = len(sig_norm)
    yf = fft(sig_norm)
    xf = fftfreq(N, 1/FS)
    pos_mask = (xf > 0.05) & (xf < 1.5) # Focus on breathing range
    xf_plot = xf[pos_mask]
    yf_plot = np.abs(yf[pos_mask])
    yf_norm = yf_plot / np.max(yf_plot)
    
    # Extract peak frequency for display
    peak_freq = xf_plot[np.argmax(yf_norm)]
    bpm_fft = peak_freq * 60
    
    time_axis = np.linspace(0, len(sig_norm)/FS, len(sig_norm))
    
    return time_axis, sig_norm, xf_plot, yf_norm, peak_freq, bpm_peaks, peaks

# Process Data
time_n, wave_n, xf_n, yf_n, freq_n, bpm_n, pks_n = get_respiratory_data(NORMAL_FILE)
time_a, wave_a, xf_a, yf_a, freq_a, bpm_a, pks_a = get_respiratory_data(ABNORMAL_FILE)

# --- Plotting ---
fig, axs = plt.subplots(2, 2, figsize=(14, 9))

# [0,0] Normal Time Domain + Peaks
axs[0, 0].plot(time_n, wave_n, color='green', label='Signal')
axs[0, 0].plot(time_n[pks_n], wave_n[pks_n], "x", color='black', label='Breaths')
axs[0, 0].set_title(f"Normal (Eupnea) - Time Domain\nEstimated BPM: {bpm_n:.1f}")
axs[0, 0].set_ylabel("Amplitude (Std Devs)")
axs[0, 0].grid(True, alpha=0.3)
axs[0, 0].legend(loc='upper right')

# [0,1] Abnormal Time Domain + Peaks
axs[0, 1].plot(time_a, wave_a, color='red', label='Signal')
axs[0, 1].plot(time_a[pks_a], wave_a[pks_a], "x", color='black', label='Breaths')
axs[0, 1].set_title(f"Abnormal (Dyspnea) - Time Domain\nEstimated BPM: {bpm_a:.1f}")
axs[0, 1].grid(True, alpha=0.3)
axs[0, 1].legend(loc='upper right')

# [1,0] Normal FFT
axs[1, 0].plot(xf_n, yf_n, color='green')
axs[1, 0].set_title("Normal FFT Spectrum")
axs[1, 0].set_xlabel("Frequency (Hz)")
axs[1, 0].set_xlim(0, 1.5)
axs[1, 0].grid(True, alpha=0.3)

# [1,1] Abnormal FFT
axs[1, 1].plot(xf_a, yf_a, color='red')
axs[1, 1].set_title("Abnormal FFT Spectrum")
axs[1, 1].set_xlabel("Frequency (Hz)")
axs[1, 1].set_xlim(0, 1.5)
axs[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()