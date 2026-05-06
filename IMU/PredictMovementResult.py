import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from scipy import signal
import os
import glob
import random
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ==========================================
# 1. Configuration (Updated for Verification)
# ==========================================
# PATH IS NOW STRICTLY FOR VERIFICATION SAMPLES
VERIFY_DATA_DIR = os.path.join('data', 'Movement Verify Samples') 
MODEL_PATH = 'models/movement_classifier.keras'
SCALER_PATH = 'models/movement_scaler.pkl'

TARGET_FREQ = 20        
WINDOW_SIZE = 200       
STEP_SIZE = 40          

# Exact column names from your provided CSV sample
FEATURES_CSV = ['AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ', 'Roll', 'Pitch', 'Yaw']
LABEL_MAP = {'Static Stand': 0, 'Slow Walk': 1, 'Fast Walk': 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

# ==========================================
# 2. Logic Functions
# ==========================================

def clean_signal_logic(data, fs=20):
    cleaned_data = np.zeros_like(data)
    nyq = 0.5 * fs
    b, a = signal.butter(2, [0.1/nyq, 9.0/nyq], btype='band')
    for i in range(data.shape[1]):
        feat_filt = signal.filtfilt(b, a, data[:, i])
        try:
            cleaned_data[:, i] = signal.savgol_filter(feat_filt, 11, 2)
        except:
            cleaned_data[:, i] = feat_filt
    return cleaned_data

# ==========================================
# 3. Main Execution
# ==========================================

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        exit()
        
    print("Loading AI Model and Scaler...")
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # 1. Collect all CSV files from VERIFY folder only
    file_list = []
    for folder_name, label_idx in LABEL_MAP.items():
        path = os.path.join(VERIFY_DATA_DIR, folder_name, "*.csv")
        files = glob.glob(path)
        for f in files:
            file_list.append((f, folder_name))
    
    if not file_list:
        print(f"❌ Error: No CSV files found in {VERIFY_DATA_DIR}")
        exit()

    random.shuffle(file_list)
    print(f"🚀 Found {len(file_list)} verification files. Starting randomized test...")

    all_y_true = []
    all_y_pred = []

    print(f"\n{'File Name':<30} | {'Actual':<12} | {'AI Pred':<12} | {'Result'}")
    print("-" * 85)

    for f_path, true_label in file_list:
        try:
            # Load CSV - Keeping exact casing
            df = pd.read_csv(f_path)
            
            # Map the specific column name 'UnixTimestamp' for resampling
            if 'UnixTimestamp' in df.columns:
                df['dt'] = pd.to_datetime(df['UnixTimestamp'], unit='ms')
            elif 'unixtimestamp' in df.columns:
                df['dt'] = pd.to_datetime(df['unixtimestamp'], unit='ms')
            else:
                # Fallback if somehow the header is missing
                df['dt'] = pd.to_datetime(df.iloc[:, 1], unit='ms')

            df_res = df.set_index('dt').resample('50ms').mean().interpolate().ffill().bfill()

            # Extract Features using the exact casing provided: AccX, AccY, etc.
            raw_feats = df_res.reindex(columns=FEATURES_CSV, fill_value=0).values.astype(np.float32)
            
            # Feature Engineering: Magnitudes (Index 0-2 for Acc, 3-5 for Gyro)
            acc_mag = np.linalg.norm(raw_feats[:, :3], axis=1, keepdims=True)
            gyro_mag = np.linalg.norm(raw_feats[:, 3:6], axis=1, keepdims=True)
            combined = np.hstack([raw_feats, acc_mag, gyro_mag]) # 11 features total
            
            cleaned = clean_signal_logic(combined, fs=TARGET_FREQ)

            file_preds = []
            confidences = []
            
            for i in range(0, len(cleaned) - WINDOW_SIZE, STEP_SIZE):
                window = cleaned[i : i + WINDOW_SIZE]
                
                # Pre-processing: Zero-centering (matching training logic)
                window_centered = window - np.mean(window, axis=0)
                
                # Scaling: Match the 11-feature input (9 raw + 2 mags)
                window_scaled = scaler.transform(window_centered.reshape(-1, 11)).reshape(1, WINDOW_SIZE, 11)
                
                probs = model.predict(window_scaled, verbose=0)[0]
                file_preds.append(np.argmax(probs))
                confidences.append(np.max(probs))

            if not file_preds:
                continue

            # Majority Vote for the whole file
            final_pred_idx = Counter(file_preds).most_common(1)[0][0]
            final_pred_name = INV_LABEL_MAP[final_pred_idx]
            avg_conf = np.mean(confidences)
            
            all_y_true.append(LABEL_MAP[true_label])
            all_y_pred.append(final_pred_idx)

            short_name = os.path.basename(f_path)[:30]
            status_icon = "✅ PASS" if final_pred_name == true_label else "❌ FAIL"
            print(f"{short_name:<30} | {true_label:<12} | {final_pred_name:<12} | {status_icon} ({avg_conf:.1%})")

        except Exception as e:
            print(f"Error processing {os.path.basename(f_path)}: {e}")

    # ==========================================
    # 4. English Statistical Report
    # ==========================================
    if all_y_true:
        print("\n" + "="*50)
        print("      RANDOMIZED VERIFICATION REPORT")
        print("="*50)
        
        print(classification_report(all_y_true, all_y_pred, target_names=LABEL_MAP.keys()))
        
        cm = confusion_matrix(all_y_true, all_y_pred)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=LABEL_MAP.keys(), yticklabels=LABEL_MAP.keys(), cmap='magma')
        plt.title('Verification Results: Confusion Matrix (Movement Verify Samples)')
        plt.ylabel('Actual Category')
        plt.xlabel('AI Prediction')
        plt.show()
    else:
        print("No data was successfully processed. Check CSV headers.")