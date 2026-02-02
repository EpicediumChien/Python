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
# 1. Configuration
# ==========================================
DATA_DIR = os.path.join('TrainingData', 'Static Sit') 
TARGET_FREQ = 25        
WINDOW_SIZE = 250   
STEP_SIZE = 50 # Maximum overlap for the largest possible dataset

# ==========================================
# 2. Advanced Feature Engineering
# ==========================================

def get_invariant_signals(df_window):
    signals = []
    # Acc/Gyro Magnitude & PCA
    for prefix in ['acc', 'gyro']:
        cols = [f'{prefix}X', f'{prefix}Y', f'{prefix}Z']
        arr = df_window[cols].values
        signals.append(np.linalg.norm(arr, axis=1)) # Magnitude
        pca = PCA(n_components=1)
        signals.append(pca.fit_transform(arr).flatten()) # Principal Axis
    
    # Orientation Change
    angles = df_window[['roll', 'pitch', 'yaw']].values
    diff_angles = np.diff(angles, axis=0, append=angles[-1:])
    signals.append(np.linalg.norm(diff_angles, axis=1))
    return signals

def extract_features(signals, fs=25):
    all_feats = []
    for sig in signals:
        # Tighter Bandpass to remove more noise (0.12Hz - 0.75Hz)
        nyq = 0.5 * fs
        b, a = signal.butter(3, [0.12/nyq, 0.75/nyq], btype='band')
        try: sig_filt = signal.filtfilt(b, a, sig)
        except: sig_filt = sig
        
        sig_norm = (sig_filt - np.mean(sig_filt)) / (np.std(sig_filt) + 1e-6)
        
        # --- Time Domain (Shape & Complexity) ---
        all_feats.append(np.std(sig_norm))
        all_feats.append(stats.skew(sig_norm))
        all_feats.append(stats.kurtosis(sig_norm))
        
        # Crest Factor (Peak-to-RMS ratio - helps find "sharp" gasping)
        crest_factor = np.max(np.abs(sig_norm)) / np.sqrt(np.mean(sig_norm**2))
        all_feats.append(crest_factor)
        
        # Autocorrelation Peak (Rhythm Stability)
        ac = np.correlate(sig_norm, sig_norm, mode='full')[len(sig_norm)-1:]
        ac_peak = np.max(ac[20:]) / (ac[0] + 1e-6)
        all_feats.append(ac_peak)
        
        # --- Frequency Domain ---
        freqs, psd = signal.welch(sig_norm, fs, nperseg=WINDOW_SIZE)
        psd_norm = psd / (np.sum(psd) + 1e-12)
        
        all_feats.append(freqs[np.argmax(psd)]) # Peak Freq
        # Spectral Entropy (High entropy = noise, Low = pure breath)
        se = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
        all_feats.append(se)
        
    return np.array(all_feats)

def time_warp_df(df, factor):
    x = np.arange(len(df))
    x_new = np.linspace(0, (len(df)-1) * factor, len(df))
    # Use cubic interpolation for smoother "synthetic" waves
    return pd.DataFrame({col: interpolate.interp1d(x, df[col].values, kind='cubic', fill_value="extrapolate")(x_new) for col in df.columns})

# ==========================================
# 3. Buffer-Zone Training Strategy
# ==========================================
if __name__ == "__main__":
    all_files = glob.glob(os.path.join(DATA_DIR, '*.csv'))
    X_list, Y_list = [], []

    print("Generating Training Set with BUFFER ZONE (19-24 BPM removed from training)...")

    for f in all_files:
        df_raw = pd.read_csv(f)[['accX', 'accY', 'accZ', 'gyroX', 'gyroY', 'gyroZ', 'roll', 'pitch', 'yaw']]
        
        for i in range(0, len(df_raw) - WINDOW_SIZE, STEP_SIZE):
            window = df_raw.iloc[i : i + WINDOW_SIZE]
            
            # --- 1. STRICT NORMAL (12-18 BPM) ---
            # Original
            X_list.append(extract_features(get_invariant_signals(window)))
            Y_list.append(0)
            # Safe Normal (Random 13-17 BPM)
            X_list.append(extract_features(get_invariant_signals(time_warp_df(window, np.random.uniform(0.85, 1.1)))))
            Y_list.append(0)

            # --- 2. CLEAR ABNORMAL (<10 or >25 BPM) ---
            # We skip the 19-24 range to teach the model "Clear Differences"
            # Very Slow (6-10 BPM)
            X_list.append(extract_features(get_invariant_signals(time_warp_df(window, np.random.uniform(0.4, 0.65)))))
            Y_list.append(1)
            # Very Fast (25-38 BPM)
            X_list.append(extract_features(get_invariant_signals(time_warp_df(window, np.random.uniform(1.7, 2.5)))))
            Y_list.append(1)

    X = np.nan_to_num(np.array(X_list))
    Y = np.array(Y_list)
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, stratify=Y, random_state=42)
    
    print(f"Training on {len(X_train)} samples...")

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=25,
        min_samples_leaf=1,
        max_features='log2', # Uses fewer features per split to force learning subtle patterns
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]
    
    print("\n=== Buffer-Zone Performance Report ===")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Abnormal']))

    print("\nStrict Precision Check (Goal: 100%):")
    for thresh in [0.5, 0.8, 0.9]:
        mask = (y_probs >= thresh) | (y_probs <= (1-thresh))
        y_strict = (y_probs[mask] >= thresh).astype(int)
        prec = precision_score(y_test[mask], y_strict, pos_label=1, zero_division=0)
        print(f"Confidence > {thresh:.1f} | Abnormal Precision: {prec:.4f}")

    joblib.dump(model, 'models/robust_breathing_model.pkl')