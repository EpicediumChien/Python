import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from scipy import signal
import os, glob, random, joblib, sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
# To stable transoformer
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import GaussianNoise
from layers_utils import PositionalEmbedding, transformer_encoder

# ==========================================
# 1. Configuration
# ==========================================
DATA_DIR = os.path.join('TrainingData', 'Static Sit')
TARGET_FREQ = 20        
WINDOW_SIZE = 200       
STEP_SIZE = 10          # High overlap (Matched)
AUGMENTATION_COUNT = 1     
FEATURES = ['accX', 'accY', 'accZ', 'gyroX', 'gyroY', 'gyroZ', 'roll', 'pitch', 'yaw']

# ==========================================
# 2. Signal Processing
# ==========================================
def apply_agc_logic(sig, fs=20, window_sec=5):
    window_size = int(fs * window_sec)
    rolling_std = pd.Series(sig).rolling(window=window_size, center=True).std()
    rolling_std = rolling_std.bfill().ffill().replace(0, 1e-6)
    return sig / rolling_std.values

def clean_signal_logic(data, fs=20):
    cleaned_data = np.zeros_like(data)
    nyq = 0.5 * fs
    
    # 1. 你的原始濾波器 [0.08Hz - 0.85Hz]
    b, a = signal.butter(2, [0.08/nyq, 0.85/nyq], btype='band')
    
    # 2. 你的原始平滑視窗 (1.1s)
    savgol_win = 27 # 25Hz * 1.1s ≈ 27
    
    for i in range(data.shape[1]):
        # A. 帶通濾波
        feat_filt = signal.filtfilt(b, a, data[:, i])
        
        # B. 你的原始 AGC (10秒視窗)
        feat_agc = apply_agc_logic(feat_filt, fs=fs, window_sec=10)
        
        # C. 你的原始 SavGol
        try:
            cleaned_data[:, i] = signal.savgol_filter(feat_agc, window_length=savgol_win, polyorder=2)
        except:
            cleaned_data[:, i] = feat_agc
            
    return cleaned_data

# ==========================================
# 3. Model Components
# ==========================================
@tf.keras.utils.register_keras_serializable(package="MyLayers")
# class PositionalEmbedding(layers.Layer):
#     def __init__(self, sequence_length, embed_dim, **kwargs):
#         # We ensure the name of the layer is what Keras expects
#         super().__init__(**kwargs)
#         self.sequence_length = sequence_length
#         self.embed_dim = embed_dim
#         # Create an internal embedding layer with the name 'embedding'
#         self.embedding = layers.Embedding(
#             input_dim=sequence_length, 
#             output_dim=embed_dim, 
#             name="embedding" 
#         )

#     def call(self, inputs):
#         # inputs shape: (Batch, 50, 128)
#         length = tf.shape(inputs)[1]
#         positions = tf.range(start=0, limit=length, delta=1)
#         embedded_positions = self.embedding(positions)
#         return inputs + embedded_positions

#     def get_config(self):
#         config = super().get_config()
#         config.update({
#             "sequence_length": self.sequence_length,
#             "embed_dim": self.embed_dim,
#         })
#         return config

# def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
#     x = layers.LayerNormalization(epsilon=1e-6)(inputs)
#     x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
#     x = layers.Dropout(dropout)(x)
#     res = x + inputs
#     x = layers.LayerNormalization(epsilon=1e-6)(res)
#     x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
#     x = layers.Dropout(dropout)(x)
#     x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
#     return x + res

def build_hybrid_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    # 1. Add Noise (The runner on sand) - CRUCIAL
    x = GaussianNoise(0.01)(inputs)

    # 2. CNN Block - Add L2 Regularization
    x = layers.Conv1D(64, kernel_size=7, padding="same", activation="relu", kernel_regularizer=l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.SpatialDropout1D(0.2)(x)
    x = layers.MaxPooling1D(pool_size=2)(x) # 200 -> 100

    x = layers.Conv1D(128, kernel_size=5, padding="same", activation="relu", kernel_regularizer=l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x) # 100 -> 50
    
    # 3. Transformer Block - Simplify it
    x = PositionalEmbedding(50, 128)(x) 
    # Use smaller head_size to prevent over-memorization
    x = transformer_encoder(x, head_size=32, num_heads=2, ff_dim=64, dropout=0.5)
    
    # 4. Hybrid Pooling
    avg_p = layers.GlobalAveragePooling1D()(x)
    max_p = layers.GlobalMaxPooling1D()(x)
    combined = layers.Concatenate()([avg_p, max_p])

    # 5. Output Head - Add L2
    x = layers.Dense(32, activation="relu", kernel_regularizer=l2(0.01))(combined)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs)
    
    # 6. Add Label Smoothing (The "LSTM Secret")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4), 
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1), 
        metrics=["accuracy"]
    )
    return model

# ==========================================
# 4. Data Generation Logic
# ==========================================
def get_dominant_bpm(window_data, fs=20):
    sig = window_data[:, 9] # Acc Mag
    sig = sig - np.mean(sig)
    freqs = np.fft.rfftfreq(len(sig), d=1/fs)
    fft_mag = np.abs(np.fft.rfft(sig))
    idx = np.where((freqs >= 0.05) & (freqs <= 1.0))[0]
    return freqs[idx][np.argmax(fft_mag[idx])] * 60 if len(idx)>0 else 15.0

def generate_abnormal_by_bpm(window_data, fs=20):
    orig_bpm = get_dominant_bpm(window_data, fs)
    orig_len = len(window_data)
    
    # 嚴格定義異常頻率
    target_bpm = random.uniform(25, 40) if random.random() > 0.5 else random.uniform(6, 10)
    
    ratio = target_bpm / orig_bpm
    new_len = max(int(orig_len / ratio), 1)
    resampled = signal.resample(window_data, new_len)
    
    # 這裡只做重複或截斷，不要做 **1.4 扭曲
    if ratio > 1.0:
        output = np.tile(resampled, (int(np.ceil(orig_len / new_len)), 1))[:orig_len, :]
    else:
        output = resampled[:orig_len, :]
        if len(output) < orig_len:
            output = np.pad(output, ((0, orig_len - len(output)), (0, 0)), mode='edge')

    # 只加入極小量的噪聲 (模擬感測器底噪)
    return output + np.random.normal(0, 0.005, output.shape)

def process_single_file(file_path):
    try:
        df = pd.read_csv(file_path)
        df.columns = [c.lower() for c in df.columns]
        ts_col = 'unix_timestamp' if 'unix_timestamp' in df.columns else 'timestamp'
        df['dt'] = pd.to_datetime(df[ts_col], unit='ms')
        df_res = df.set_index('dt').resample('50ms').mean().interpolate().ffill().bfill()

        acc_mag = np.linalg.norm(df_res[[c for c in df_res.columns if 'acc' in c][:3]].values, axis=1, keepdims=True)
        gyro_mag = np.linalg.norm(df_res[[c for c in df_res.columns if 'gyro' in c][:3]].values, axis=1, keepdims=True)
        raw_data = df_res.reindex(columns=[f.lower() for f in FEATURES]).values.astype(np.float32)
        combined = np.hstack([raw_data, acc_mag, gyro_mag]) 
        processed_data = clean_signal_logic(combined, fs=TARGET_FREQ)

        x_list, y_list = [], []
        for i in range(0, len(processed_data) - WINDOW_SIZE, STEP_SIZE):
            window = processed_data[i : i + WINDOW_SIZE]
            bpm = get_dominant_bpm(window, fs=TARGET_FREQ)
            
            # Per-Window Normalization (Aligned)
            window_norm = (window - np.mean(window, axis=0)) / (np.std(window, axis=0) + 1e-6)

            if 12.0 <= bpm <= 20.0:
                x_list.append(window_norm); y_list.append(0)
                for _ in range(AUGMENTATION_COUNT):
                    ab_win = generate_abnormal_by_bpm(window, fs=TARGET_FREQ)
                    ab_norm = (ab_win - np.mean(ab_win, axis=0)) / (np.std(ab_win, axis=0) + 1e-6)
                    x_list.append(ab_norm); y_list.append(1)
            else:
                x_list.append(window_norm); y_list.append(1)
        return np.array(x_list), np.array(y_list)
    except: return None, None

# ==========================================
# 5. Training Execution
# ==========================================
if __name__ == "__main__":
    file_list = glob.glob(os.path.join(DATA_DIR, '*.csv'))
    random.shuffle(file_list)
    
    def collect_and_balance(files):
        x_raw, y_raw = [], []
        for f in files:
            xf, yf = process_single_file(f)
            if xf is not None: x_raw.append(xf); y_raw.append(yf)
        X, y = np.concatenate(x_raw), np.concatenate(y_raw)
        idx0, idx1 = np.where(y == 0)[0], np.where(y == 1)[0]
        min_s = min(len(idx0), len(idx1))
        balanced_idx = np.concatenate([idx0[:min_s], idx1[:min_s]])
        np.random.shuffle(balanced_idx)
        return X[balanced_idx], y[balanced_idx]

    split = int(len(file_list) * 0.8)
    X_train, y_train = collect_and_balance(file_list[:split])
    X_test, y_test = collect_and_balance(file_list[split:])

    model = build_hybrid_model((WINDOW_SIZE, 11))
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=64,
                        callbacks=[EarlyStopping(patience=15, restore_best_weights=True), ReduceLROnPlateau(patience=5)])

    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.4).astype(int)
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Abnormal']))

    # 2. Classification Report
    print("\n[Classification Report]")
    target_names = ['Normal (Eupnea)', 'Abnormal (Dyspnea/Apnea)']
    print(classification_report(y_test, y_pred, target_names=target_names))

    # 3. Plotting Training History
    plt.figure(figsize=(12, 5))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 4. Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()

    # 5. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()

    # ==========================================
    # 6. Save Model Artifacts
    # ==========================================
    os.makedirs('models', exist_ok=True)
    model.save('models/respiration_CNNxTran_classifier.keras')
    # joblib.dump(scaler, 'models/CNNxTran_scaler.pkl')
    print(f"\nTraining Complete. Model saved to 'models/respiration_CNNxTran_classifier.keras' (AUC: {roc_auc:.4f})")