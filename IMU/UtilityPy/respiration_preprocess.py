"""Signal processing, feature engineering, and window normalization."""

import pandas as pd
import numpy as np
from scipy import signal

TARGET_FREQ = 20
WINDOW_SIZE = 200
STEP_SIZE = 10
FEATURES = [
    'accX', 'accY', 'accZ',
    'gyroX', 'gyroY', 'gyroZ',
    'roll', 'pitch', 'yaw',
]
SAVGOL_WIN = 15
WINDOW_NOISE_STD = 0.002
NUM_FEATURES = len(FEATURES) + 2  # acc/gyro magnitude


def apply_agc_logic(sig, fs=20, window_sec=10):
    window_size = int(fs * window_sec)
    rolling_std = pd.Series(sig).rolling(window=window_size, center=True).std()
    rolling_std = rolling_std.bfill().ffill().replace(0, 1e-6)
    return sig / rolling_std.values


def clean_signal_logic(data, fs=20, savgol_win=SAVGOL_WIN):
    cleaned_data = np.zeros_like(data)
    nyq = 0.5 * fs
    b, a = signal.butter(2, [0.08 / nyq, 0.85 / nyq], btype='band')

    for i in range(data.shape[1]):
        feat_filt = signal.filtfilt(b, a, data[:, i])
        feat_agc = apply_agc_logic(feat_filt, fs=fs, window_sec=10)
        try:
            cleaned_data[:, i] = signal.savgol_filter(
                feat_agc, window_length=savgol_win, polyorder=2
            )
        except Exception:
            cleaned_data[:, i] = feat_agc

    return cleaned_data


def load_resampled_dataframe(file_path):
    df = pd.read_csv(file_path)
    df.columns = [c.lower() for c in df.columns]
    if len(df) < 50:
        return None

    ts_col = next(c for c in ['unix_timestamp', 'timestamp'] if c in df.columns)
    df['dt'] = pd.to_datetime(df[ts_col], unit='ms')
    df_res = df.set_index('dt').resample('50ms').mean().interpolate().ffill().bfill()
    if len(df_res) <= 20:
        return None
    return df_res


def build_feature_matrix(df_res):
    acc_cols = [c for c in df_res.columns if 'acc' in c][:3]
    gyro_cols = [c for c in df_res.columns if 'gyro' in c][:3]
    acc_mag = np.linalg.norm(df_res[acc_cols].values, axis=1, keepdims=True)
    gyro_mag = np.linalg.norm(df_res[gyro_cols].values, axis=1, keepdims=True)
    raw_data = df_res.reindex(columns=[f.lower() for f in FEATURES]).values.astype(np.float32)
    return np.hstack([raw_data, acc_mag, gyro_mag])


def load_processed_file(file_path, fs=TARGET_FREQ, window_size=WINDOW_SIZE):
    """Load CSV, resample, engineer features, and return cleaned signal array."""
    try:
        df_res = load_resampled_dataframe(file_path)
        if df_res is None:
            return None

        combined = build_feature_matrix(df_res)
        processed_data = clean_signal_logic(combined, fs=fs)
        if len(processed_data) < window_size:
            return None
        return processed_data
    except Exception as e:
        print(f"Error in {file_path}: {e}")
        return None


def add_window_noise(window, std=WINDOW_NOISE_STD):
    return window + np.random.normal(0, std, window.shape)


def normalize_window(window):
    return (window - np.mean(window, axis=0)) / (np.std(window, axis=0) + 1e-6)


def iter_windows(processed_data, window_size=WINDOW_SIZE, step_size=STEP_SIZE):
    for i in range(0, len(processed_data) - window_size, step_size):
        yield processed_data[i:i + window_size]
