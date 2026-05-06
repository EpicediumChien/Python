import os
import glob
import pandas as pd
import numpy as np
import joblib
from scipy import signal, fft, interpolate, stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import PCA

# ==========================================
# 1. Configuration
# ==========================================
DATA_DIR = os.path.join('TrainingData', 'Static Sit') 
TARGET_FREQ = 25        
WINDOW_SIZE = 250   
STEP_SIZE = 40 # Maximum density for training data

# ==========================================
# 2. Advanced Feature Extraction
# ==========================================

def clean_signal_original_logic(data, fs=25):
    nyq = 0.5 * fs
    low, high = 0.1 / nyq, 0.8 / nyq 
    b, a = signal.butter(2, [low, high], btype='band')
    feat = signal.medfilt(data, kernel_size=5)
    try: cleaned_data = signal.filtfilt(b, a, feat)
    except: cleaned_data = feat
    return cleaned_data

def get_invariant_signals(df_window):
    signals = []
    # Acc/Gyro Magnitude & PCA
    for prefix in ['acc', 'gyro']:
        arr = df_window[[f'{prefix}X', f'{prefix}Y', f'{prefix}Z']].values
        signals.append(np.linalg.norm(arr, axis=1)) # Magnitude
        signals.append(PCA(n_components=1).fit_transform(arr).flatten()) # Principal Axis
    
    # Orientation Change
    angles = df_window[['roll', 'pitch', 'yaw']].values
    diff_angles = np.diff(angles, axis=0, append=angles[-1:])
    signals.append(np.linalg.norm(diff_angles, axis=1))
    return signals

def extract_features(signals, fs=25):
    all_feats = []
    
    # Feature 1: Cross-Correlation between Acc and Gyro Magnitude (Synchronization)
    acc_mag = signals[0]
    gyro_mag = signals[2]
    corr = np.corrcoef(acc_mag, gyro_mag)[0, 1]
    all_feats.append(corr if not np.isnan(corr) else 0)

    for sig in signals:
        sig_cleaned = clean_signal_original_logic(sig, fs=fs)
        sig_norm = (sig_cleaned - np.mean(sig_cleaned)) / (np.std(sig_cleaned) + 1e-6)
        
        # --- Time Domain Shape ---
        all_feats.extend([np.std(sig_norm), stats.skew(sig_norm), stats.kurtosis(sig_norm)])
        
        # --- Hilbert Envelope Stability ---
        analytic_signal = signal.hilbert(sig_norm)
        amplitude_envelope = np.abs(analytic_signal)
        envelope_stability = np.std(amplitude_envelope) / (np.mean(amplitude_envelope) + 1e-6)
        all_feats.append(envelope_stability)
        
        # --- Frequency Domain (Sub-band analysis) ---
        freqs, psd = signal.welch(sig_norm, fs, nperseg=125, noverlap=62, nfft=1024)
        
        # Peak Freq
        peak_idx = np.argmax(psd)
        all_feats.append(freqs[peak_idx])
        
        # Energy Ratio: Normal (0.1-0.35Hz) vs Abnormal (0.35-0.8Hz)
        e_normal = np.sum(psd[(freqs >= 0.1) & (freqs <= 0.35)])
        e_abnormal = np.sum(psd[(freqs > 0.35) & (freqs <= 0.8)])
        all_feats.append(e_normal / (e_abnormal + 1e-12))
        
        # Spectral Flatness & Autocorrelation
        spec_flatness = stats.gmean(psd + 1e-12) / (np.mean(psd) + 1e-12)
        ac = np.correlate(sig_norm, sig_norm, mode='full')[len(sig_norm)-1:]
        ac_peak = np.max(ac[30:150]) / (ac[0] + 1e-6)
        
        all_feats.extend([spec_flatness, ac_peak])
        
    return np.array(all_feats)

def time_warp_df(df, factor):
    x = np.arange(len(df))
    x_new = np.linspace(0, (len(df)-1) * factor, len(df))
    warped = pd.DataFrame({col: interpolate.interp1d(x, df[col].values, kind='cubic', fill_value="extrapolate")(x_new) for col in df.columns})
    
    # Inject 1% Synthetic Noise to prevent "perfect signal" bias
    noise = np.random.normal(0, 0.01, warped.shape)
    return warped + noise

# ==========================================
# 3. Execution (Massive Augmentation)
# ==========================================
if __name__ == "__main__":
    all_files = glob.glob(os.path.join(DATA_DIR, '*.csv'))
    X_list, Y_list = [], []

    print("Generating Multi-Domain Fusion dataset...")

    for f in all_files:
        df_raw = pd.read_csv(f)[['accX', 'accY', 'accZ', 'gyroX', 'gyroY', 'gyroZ', 'roll', 'pitch', 'yaw']]
        
        for i in range(0, len(df_raw) - WINDOW_SIZE, STEP_SIZE):
            window = df_raw.iloc[i : i + WINDOW_SIZE]
            
            # --- NORMAL (3 Variations) ---
            X_list.append(extract_features(get_invariant_signals(window))) # Original
            Y_list.append(0)
            X_list.append(extract_features(get_invariant_signals(time_warp_df(window, 0.85)))) # 12.75 BPM
            Y_list.append(0)
            X_list.append(extract_features(get_invariant_signals(time_warp_df(window, 1.2)))) # 18 BPM
            Y_list.append(0)

            # --- ABNORMAL (3 Variations) ---
            X_list.append(extract_features(get_invariant_signals(time_warp_df(window, 0.5))))  # 7.5 BPM
            Y_list.append(1)
            X_list.append(extract_features(get_invariant_signals(time_warp_df(window, 1.8))))  # 27 BPM
            Y_list.append(1)
            X_list.append(extract_features(get_invariant_signals(time_warp_df(window, 2.5))))  # 37.5 BPM
            Y_list.append(1)

    X = np.nan_to_num(np.array(X_list))
    Y = np.array(Y_list)
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, stratify=Y, random_state=42)
    
    # Using XGBoost-style depth for Random Forest
    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=15, 
        max_features='sqrt',
        min_samples_leaf=1,
        bootstrap=True,
        class_weight='balanced', 
        random_state=42, 
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print("\n=== VERSION 12: MULTI-DOMAIN FUSION RESULTS ===")
    print(classification_report(y_test, y_pred, digits=4))
    print(f"Final Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    # Top Feature Importance Check
    importances = model.feature_importances_
    print(f"Total Features used: {len(importances)}")

    joblib.dump(model, 'models/robust_breathing_model.pkl')