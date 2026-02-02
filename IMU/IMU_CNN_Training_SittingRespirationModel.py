import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from scipy import signal
import os
import glob
import random
import joblib
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# --- GPU MEMORY MANAGEMENT ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU detected: {len(gpus)} device(s) ready.")
    except Exception as e:
        print(f"GPU Setup error: {e}")

# ==========================================
# 1. Configuration
# ==========================================
DATA_DIR = os.path.join('TrainingData', 'Static Sit')
TARGET_FREQ = 25        
WINDOW_SIZE = 150       
STEP_SIZE = 30          
AUGMENTATION_COUNT = 5  
FEATURES = ['accX', 'accY', 'accZ', 'gyroX', 'gyroY', 'gyroZ', 'roll', 'pitch', 'yaw']

# ==========================================
# 2. Pre-processing Logic
# ==========================================
def clean_signal_logic(data, fs=25):
    # Safety Check: If data is too short for the kernel, skip filtering
    kernel = int(fs * 0.12)
    if kernel % 2 == 0: kernel += 1
    if len(data) <= kernel: return data 

    cleaned_data = np.zeros_like(data)
    nyq = 0.5 * fs
    low, high = 0.05 / nyq, 0.8 / nyq 
    b, a = signal.butter(4, [low, high], btype='band')
    
    for i in range(data.shape[1]):
        feat = signal.medfilt(data[:, i], kernel_size=kernel)
        try:
            cleaned_data[:, i] = signal.filtfilt(b, a, feat)
        except:
            cleaned_data[:, i] = feat
    return cleaned_data

def get_random_abnormal_factor():
    rand_val = random.random()
    if rand_val < 0.25: return 0.0  # Apnea
    elif rand_val < 0.6: return random.uniform(0.1, 0.65) # Phase 3: Bradypnea
    else: return random.uniform(1.45, 3.5) # Phase 2: Tachypnea

def generate_abnormal_sample(window_data, factor):
    orig_len = len(window_data)
    if factor == 0.0: return np.random.normal(0, 0.02, window_data.shape)
    new_len = max(int(orig_len / factor), 1)
    resampled = signal.resample(window_data, new_len)
    output = np.tile(resampled, (orig_len // new_len + 1, 1))[:orig_len, :]
    return output + np.random.normal(0, 0.005, output.shape)

def process_single_file(file_path):
    try:
        df = pd.read_csv(file_path)
        ts_col = [c for c in df.columns if 'timestamp' in c.lower()]
        if not ts_col: return None, None
        
        raw_ts = pd.to_numeric(df[ts_col[0]], errors='coerce').values
        raw_ts = raw_ts[~np.isnan(raw_ts)]
        if len(raw_ts) < WINDOW_SIZE: return None, None # Skip early if too short
        
        relative_ts = raw_ts - raw_ts[0]
        unit = 'us' if abs(raw_ts[-1]-raw_ts[0]) > 1000000 else 'ms' if abs(raw_ts[-1]-raw_ts[0]) > 1000 else 's'
        
        df = df.iloc[:len(raw_ts)].copy()
        df['datetime'] = pd.to_datetime(relative_ts, unit=unit, origin=pd.Timestamp('2000-01-01'))
        df = df.sort_values('datetime').drop_duplicates(subset=['datetime']).set_index('datetime')
        
        df_res = df[FEATURES].resample("40ms").mean().interpolate().ffill().bfill()
        data = df_res.values.astype(np.float32)
        
        # Add magnitudes
        data = np.hstack([data, np.linalg.norm(data[:, :3], axis=1, keepdims=True), np.linalg.norm(data[:, 3:6], axis=1, keepdims=True)]) 
        
        # Filter AFTER resampling and length check to avoid warnings
        data = clean_signal_logic(data, fs=TARGET_FREQ)
        
        if len(data) < WINDOW_SIZE: return None, None
        
        x_list, y_list = [], []
        for i in range(0, len(data) - WINDOW_SIZE, STEP_SIZE):
            window = data[i : i + WINDOW_SIZE]
            window = (window - np.mean(window, axis=0)) / (np.std(window, axis=0) + 1e-6)
            x_list.append(window); y_list.append(0) 
            for _ in range(AUGMENTATION_COUNT):
                f = get_random_abnormal_factor()
                x_list.append(generate_abnormal_sample(window, f)); y_list.append(1)
        return np.array(x_list), np.array(y_list)
    except: return None, None

# ==========================================
# 4. Model Architecture
# ==========================================
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def build_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(64, 11, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(128, 5, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    positions = tf.range(start=0, limit=x.shape[1], delta=1)
    pos_enc = layers.Embedding(input_dim=x.shape[1], output_dim=x.shape[2])(positions)
    x = x + pos_enc
    x = transformer_encoder(x, 128, 8, 256, 0.1)
    x = transformer_encoder(x, 128, 8, 256, 0.1)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.0005), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ==========================================
# 5. Plotting Function
# ==========================================
def evaluate_and_plot(history, model, X_test, y_test):
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)
    plt.figure(figsize=(16, 10))
    plt.subplot(2, 2, 1); plt.plot(history.history['accuracy'], label='Train'); plt.plot(history.history['val_accuracy'], label='Val'); plt.title('Accuracy'); plt.legend(); plt.grid(True)
    plt.subplot(2, 2, 2); plt.plot(history.history['loss'], label='Train'); plt.plot(history.history['val_loss'], label='Val'); plt.title('Loss'); plt.legend(); plt.grid(True)
    plt.subplot(2, 2, 3); cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Abnormal'], yticklabels=['Normal', 'Abnormal'])
    plt.title('Confusion Matrix')
    plt.subplot(2, 2, 4); fpr, tpr, _ = roc_curve(y_test, y_pred_prob); plt.plot(fpr, tpr, color='orange', label=f'AUC={auc(fpr, tpr):.2f}')
    plt.plot([0,1],[0,1], 'k--'); plt.title('ROC Curve'); plt.legend(); plt.grid(True)
    if not os.path.exists('plots'): os.makedirs('plots')
    plt.tight_layout(); plt.savefig('plots/final_performance.png'); plt.show()
    print(classification_report(y_test, y_pred))

# ==========================================
# 6. Main Execution
# ==========================================
if __name__ == "__main__":
    file_list = glob.glob(os.path.join(DATA_DIR, '*.csv'))
    if not file_list: print("No CSVs found."); sys.exit()
    random.shuffle(file_list)
    
    def collect(files):
        x_all, y_all = [], []
        for f in files:
            res = process_single_file(f)
            if res[0] is not None and len(res[0]) > 0: x_all.append(res[0]); y_all.append(res[1])
        return (np.concatenate(x_all), np.concatenate(y_all)) if x_all else (None, None)

    print("Preparing training data...")
    X_train_raw, y_train = collect(file_list[:int(len(file_list)*0.8)])
    X_test_raw, y_test = collect(file_list[int(len(file_list)*0.8):])
    
    if X_train_raw is None: print("No data collected."); sys.exit()

    scaler = StandardScaler()
    F = X_train_raw.shape[2]
    X_train = scaler.fit_transform(X_train_raw.reshape(-1, F)).reshape(-1, WINDOW_SIZE, F)
    X_test = scaler.transform(X_test_raw.reshape(-1, F)).reshape(-1, WINDOW_SIZE, F)
    
    # FIXED: Convert classes to numpy array to satisfy sklearn requirements
    cw = class_weight.compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
    
    model = build_model((WINDOW_SIZE, F))
    history = model.fit(X_train, y_train, epochs=150, batch_size=64, validation_data=(X_test, y_test),
                        class_weight={0:cw[0], 1:cw[1]}, callbacks=[EarlyStopping(patience=15, restore_best_weights=True), ReduceLROnPlateau(patience=5)])
    
    if not os.path.exists('models'): os.makedirs('models')
    model.save('models/breathing_transformer_gpu.keras'); joblib.dump(scaler, 'models/scaler.pkl')
    evaluate_and_plot(history, model, X_test, y_test)