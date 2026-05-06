import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from scipy import signal
import os
import glob
import random
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ==========================================
# 1. Configuration (Updated for 20Hz)
# ==========================================
BASE_DATA_DIR = os.path.join('data', 'Movement Classification')
LABEL_MAP = {'Static Stand': 0, 'Slow Walk': 1, 'Fast Walk': 2}

TARGET_FREQ = 20        # Adjusted to 20Hz
WINDOW_SIZE = 200       # 10 Seconds (20Hz * 10s = 200 samples)
STEP_SIZE = 40          # 2 Second overlap (Step every 2 seconds)
FEATURES = ['accx', 'accy', 'accz', 'gyrox', 'gyroy', 'gyroz', 'roll', 'pitch', 'yaw']

# ==========================================
# 2. Signal Processing (Nyquist Adjusted)
# ==========================================
def clean_signal_logic(data, fs=20):
    cleaned_data = np.zeros_like(data)
    nyq = 0.5 * fs # 10Hz
    
    # Bandpass filter (0.1Hz to 9Hz - keeping it below Nyquist 10Hz)
    b, a = signal.butter(2, [0.1/nyq, 9.0/nyq], btype='band')
    
    for i in range(data.shape[1]):
        feat_filt = signal.filtfilt(b, a, data[:, i])
        try:
            # SavGol window must be odd. 11 is roughly 0.5s at 20Hz
            cleaned_data[:, i] = signal.savgol_filter(feat_filt, window_length=11, polyorder=2)
        except:
            cleaned_data[:, i] = feat_filt
    return cleaned_data

def process_single_file(file_path, label):
    try:
        df = pd.read_csv(file_path)
        df.columns = [c.lower().strip() for c in df.columns]
        
        # Resample logic (Ensuring 50ms intervals for 20Hz)
        ts_col = 'unix_timestamp' if 'unix_timestamp' in df.columns else 'timestamp'
        df['dt'] = pd.to_datetime(df[ts_col], unit='ms')
        df = df.set_index('dt')
        df_res = df.resample('50ms').mean().interpolate().ffill().bfill()

        # Build feature matrix
        raw_data = df_res.reindex(columns=FEATURES, fill_value=0).values.astype(np.float32)
        
        # Calculate Magnitudes
        acc_mag = np.linalg.norm(raw_data[:, :3], axis=1, keepdims=True)
        gyro_mag = np.linalg.norm(raw_data[:, 3:6], axis=1, keepdims=True)
        combined_data = np.hstack([raw_data, acc_mag, gyro_mag]) # 11 features total

        processed_data = clean_signal_logic(combined_data, fs=TARGET_FREQ)

        x_list, y_list = [], []
        # Sliding window
        for i in range(0, len(processed_data) - WINDOW_SIZE, STEP_SIZE):
            window = processed_data[i : i + WINDOW_SIZE]
            # Zero-center the window (Crucial for orientation independence)
            window = window - np.mean(window, axis=0) 
            x_list.append(window)
            y_list.append(label)
                
        return np.array(x_list), np.array(y_list)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None

# ==========================================
# 3. Training Loop with File-Level Split
# ==========================================
if __name__ == "__main__":
    train_files, test_files = [], []
    
    # 1. Honest Split: Shuffle whole files, not windows
    for folder_name, label_idx in LABEL_MAP.items():
        folder_path = os.path.join(BASE_DATA_DIR, folder_name)
        files = glob.glob(os.path.join(folder_path, "*.csv"))
        random.shuffle(files)
        
        split_idx = int(0.8 * len(files))
        train_files.extend([(f, label_idx) for f in files[:split_idx]])
        test_files.extend([(f, label_idx) for f in files[split_idx:]])

    def collect_data(file_list):
        X, Y = [], []
        for f, label in file_list:
            xf, yf = process_single_file(f, label)
            if xf is not None and len(xf) > 0:
                X.append(xf); Y.append(yf)
        return np.concatenate(X), np.concatenate(Y)

    print(f"Processing 20Hz Data (Window: {WINDOW_SIZE} samples)...")
    X_train_raw, y_train = collect_data(train_files)
    X_test_raw, y_test = collect_data(test_files)

    # 2. Global Scaling
    scaler = StandardScaler()
    num_f = X_train_raw.shape[2]
    scaler.fit(X_train_raw.reshape(-1, num_f))
    
    X_train = scaler.transform(X_train_raw.reshape(-1, num_f)).reshape(-1, WINDOW_SIZE, num_f)
    X_test = scaler.transform(X_test_raw.reshape(-1, num_f)).reshape(-1, WINDOW_SIZE, num_f)

    # 3. Model Architecture (Optimized for 20Hz)
    model = models.Sequential([
        layers.Input(shape=(WINDOW_SIZE, num_f)),
        layers.Conv1D(64, 7, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Conv1D(128, 5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(len(LABEL_MAP), activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    print("Starting Training...")
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test),
              callbacks=[EarlyStopping(patience=10, restore_best_weights=True), 
                         ReduceLROnPlateau(patience=5)])

    # 4. Save Artifacts
    os.makedirs('models', exist_ok=True)
    model.save('models/movement_classifier.keras')
    joblib.dump(scaler, 'models/movement_scaler.pkl')

    # 5. Evaluation & Confidence Check
    LABEL_MAP_INV = {v: k for k, v in LABEL_MAP.items()}
    probs = model.predict(X_test)
    
    print("\n" + "="*50)
    print("20Hz RELIABILITY INSPECTION (Sample of 10)")
    print("="*50)
    
    test_indices = random.sample(range(len(X_test)), 10)
    for i in test_indices:
        pred_idx = np.argmax(probs[i])
        confidence = probs[i][pred_idx] * 100
        status = "✅" if int(y_test[i]) == pred_idx else "❌"
        print(f"{status} Actual: {LABEL_MAP_INV[int(y_test[i])]:12s} | Pred: {LABEL_MAP_INV[pred_idx]:12s} | Conf: {confidence:6.2f}%")

    # Confusion Matrix
    y_pred = np.argmax(probs, axis=1)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=LABEL_MAP.keys(), yticklabels=LABEL_MAP.keys(), cmap='viridis')
    plt.title('20Hz Movement Confusion Matrix (File-Level Split)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()