import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import pearsonr

# ==========================================
# 1. Configuration & Path Setup
# ==========================================
# Update to your specific folder
DATA_DIR = r"C:\Git\Python\IMU\data\Time Example"
FS = 20  # Change to 20 if your IMU is set to 20Hz

# Define the Metronome BPMs you tested and their corresponding filenames
# Note: Ensure the filenames match the files in your "Time Example" folder
test_cases = {
    10: "10 BPM.csv",
    14: "14 BPM.csv",
    18: "18 BPM.csv",
    22: "22 BPM.csv",
    26: "26 BPM.csv",
    30: "30 BPM.csv",
    34: "34 BPM.csv",
    38: "38 BPM.csv",
    42: "42 BPM.csv",
    46: "46 BPM.csv"
}

# ==========================================
# 2. Core Processing Function
# ==========================================
def calculate_imu_bpm(file_path, fs):
    if not os.path.exists(file_path):
        print(f"Warning: File not found {file_path}")
        return np.nan
    
    df = pd.read_csv(file_path)
    # Use Accel Magnitude (columns 3, 4, 5)
    raw_sig = np.linalg.norm(df.iloc[:, [3, 4, 5]].values, axis=1)
    
    # Simple Pre-processing for FFT
    sig_detrend = signal.detrend(raw_sig)
    nyq = 0.5 * fs
    b, a = signal.butter(2, [0.1/nyq, 1.2/nyq], btype='band') # Broad range for all tests
    sig_filt = signal.filtfilt(b, a, sig_detrend)
    
    # FFT to find dominant frequency
    N = len(sig_filt)
    yf = np.abs(np.fft.rfft(sig_filt))
    xf = np.fft.rfftfreq(N, 1/fs)
    
    # Find peak frequency in the range 0.1Hz to 1.0Hz
    idx = np.where((xf >= 0.1) & (xf <= 1.0))[0]
    peak_freq = xf[idx][np.argmax(yf[idx])]
    
    return peak_freq * 60

# ==========================================
# 3. Main Loop & Statistics
# ==========================================
results = []

for met_bpm, filename in test_cases.items():
    file_path = os.path.join(DATA_DIR, filename)
    estimated_bpm = calculate_imu_bpm(file_path, FS)
    results.append([met_bpm, estimated_bpm])

# Convert to DataFrame
df_results = pd.DataFrame(results, columns=['Metronome BPM (X)', 'IMU Estimated BPM (Y)'])
df_results = df_results.dropna() # Remove missing files

# Calculate Pearson Correlation
r_value, _ = pearsonr(df_results['Metronome BPM (X)'], df_results['IMU Estimated BPM (Y)'])

# ==========================================
# 4. Generate Result Table Image
# ==========================================
fig, ax = plt.subplots(figsize=(6, 8))
ax.axis('off')

# Prepare data for table
table_data = []
for _, row in df_results.iterrows():
    table_data.append([int(row[0]), f"{row[1]:.2f}"])

# Add correlation row
table_data.append(["Pearson Correla-\ntion coefficient (r)", f"{r_value:.10f}"])

# Create Table
table = ax.table(
    cellText=table_data,
    colLabels=df_results.columns,
    cellLoc='center',
    loc='center'
)

table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 2.5) # Scale for better readability

plt.title("IMU Accuracy Validation Result", pad=20, fontsize=14, fontweight='bold')
plt.show()

# Optional: Print to console
print(df_results)
print(f"\nPearson Correlation coefficient (r): {r_value:.10f}")