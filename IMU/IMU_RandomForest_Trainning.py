import os
import glob
import pandas as pd
import numpy as np
import joblib
from scipy import signal, fft, interpolate, stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score
from sklearn.decomposition import PCA

# ==========================================
# 1. Configuration (Matching your first code)
# ==========================================
DATA_DIR = os.path.join('TrainingData', 'Static Sit') 
TARGET_FREQ = 25        
WINDOW_SIZE = 250   
STEP_SIZE = 60 

# ==========================================
# 2. The ORIGINAL Pre-Processing Logic
# ==========================================

def clean_signal_original_logic(data, fs=25):
    """ The exact logic from your FigureForEupneaAndDyspnea.py """
    nyq = 0.5 * fs
    low, high = 0.1 / nyq, 0.8 / nyq 
    b, a = signal.butter(2, [low, high], btype='band')
    
    # Median filter (去除突波)
    kernel = 5 # fs * 0.2
    
    # Pre-process
    feat = signal.medfilt(data, kernel_size=kernel)
    try:
        cleaned_data = signal.filtfilt(b, a, feat)
    except:
        cleaned_data = feat
    return cleaned_data

def get_invariant_signals(df_window):
    """ Extracts signals that don't change when the IMU is rotated """
    signals = []
    # 1. Acc Magnitude
    acc = df_window[['accX', 'accY', 'accZ']].values
    signals.append(np.linalg.norm(acc, axis=1))
    
    # 2. Acc PCA (The main axis of chest expansion)
    pca = PCA(n_components=1)
    signals.append(pca.fit_transform(acc).flatten())
    
    # 3. Gyro Magnitude
    gyro = df_window[['gyroX', 'gyroY', 'gyroZ']].values
    signals.append(np.linalg.norm(gyro, axis=1))

    # 4. Orientation Change (Magnitude of R/P/Y tilt)
    angles = df_window[['roll', 'pitch', 'yaw']].values
    diff_angles = np.diff(angles, axis=0, append=angles[-1:])
    signals.append(np.linalg.norm(diff_angles, axis=1))
    
    return signals

def extract_features(signals, fs=25):
    all_feats = []
    for sig in signals:
        # Step A: Apply YOUR original cleaning logic
        sig_cleaned = clean_signal_original_logic(sig, fs=fs)
        
        # Step B: Standardize (AGC logic)
        sig_norm = (sig_cleaned - np.mean(sig_cleaned)) / (np.std(sig_cleaned) + 1e-6)
        
        # --- Time Domain ---
        all_feats.append(np.std(sig_norm))
        all_feats.append(stats.skew(sig_norm))
        all_feats.append(stats.kurtosis(sig_norm))
        
        # --- [CRITICAL] Frequency Domain with Hanning Window ---
        # A Hanning window prevents edge noise from creating fake frequency peaks
        win = signal.windows.hann(len(sig_norm))
        sig_win = sig_norm * win
        
        freqs, psd = signal.welch(sig_win, fs, nperseg=len(sig_norm))
        peak_freq = freqs[np.argmax(psd)]
        all_feats.append(peak_freq) # The 0.25Hz vs 0.5Hz feature
        all_feats.append(np.sum(psd))
        
        # Rhythm Strength (Autocorrelation)
        ac = np.correlate(sig_norm, sig_norm, mode='full')[len(sig_norm)-1:]
        all_feats.append(np.max(ac[20:]) / (ac[0] + 1e-6))
        
    return np.array(all_feats)

def time_warp_df(df, factor):
    """ Smooth interpolation for synthetic BPMs """
    x = np.arange(len(df))
    x_new = np.linspace(0, (len(df)-1) * factor, len(df))
    return pd.DataFrame({col: interpolate.interp1d(x, df[col].values, kind='cubic', fill_value="extrapolate")(x_new) for col in df.columns})

# ==========================================
# 3. Training Execution
# ==========================================
if __name__ == "__main__":
    all_files = glob.glob(os.path.join(DATA_DIR, '*.csv'))
    X_list, Y_list = [], []

    print(f"Applying Original Pre-processing to {len(all_files)} files...")

    for f in all_files:
        df_raw = pd.read_csv(f)[['accX', 'accY', 'accZ', 'gyroX', 'gyroY', 'gyroZ', 'roll', 'pitch', 'yaw']]
        
        for i in range(0, len(df_raw) - WINDOW_SIZE, STEP_SIZE):
            window = df_raw.iloc[i : i + WINDOW_SIZE]
            
            # --- BALANCE 1:1 ---
            # Normal (12-18 BPM)
            X_list.append(extract_features(get_invariant_signals(window)))
            Y_list.append(0)
            X_list.append(extract_features(get_invariant_signals(time_warp_df(window, 1.15)))) # 17 BPM
            Y_list.append(0)

            # Abnormal (Clear Slow / Clear Fast)
            X_list.append(extract_features(get_invariant_signals(time_warp_df(window, 0.5)))) # 7.5 BPM
            Y_list.append(1)
            X_list.append(extract_features(get_invariant_signals(time_warp_df(window, 1.8)))) # 27 BPM
            Y_list.append(1)

    X = np.nan_to_num(np.array(X_list))
    Y = np.array(Y_list)
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)
    
    # Higher quality forest
    model = RandomForestClassifier(n_estimators=500, max_depth=20, class_weight='balanced', random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print("\n=== PERFORMANCE REPORT (STRICT PRE-PROCESS) ===")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Abnormal']))

    # Check Precision
    y_probs = model.predict_proba(X_test)[:, 1]
    for thresh in [0.5, 0.8]:
        mask = (y_probs >= thresh) | (y_probs <= (1-thresh))
        prec = precision_score(y_test[mask], (y_probs[mask] >= thresh).astype(int), pos_label=1)
        print(f"Confidence > {thresh:.1f} | Abnormal Precision: {prec:.4f}")

    joblib.dump(model, 'models/robust_breathing_model.pkl')