"""BPM-based labeling and synthetic abnormal augmentation."""

import random

import numpy as np
from scipy import signal

from UtilityPy.respiration_preprocess import (
    TARGET_FREQ,
    WINDOW_SIZE,
    STEP_SIZE,
    add_window_noise,
    iter_windows,
    load_processed_file,
    normalize_window,
)

AUGMENTATION_COUNT = 1
BPM_NORMAL_MIN = 10.0
BPM_NORMAL_MAX = 22.0


def get_dominant_bpm(window_data, fs=20):
    """Detect BPM using accelerometer Z (strongest axis for sitting/lying)."""
    sig = window_data[:, 2]
    sig = sig - np.mean(sig)
    freqs = np.fft.rfftfreq(len(sig), d=1 / fs)
    fft_mag = np.abs(np.fft.rfft(sig))
    idx = np.where((freqs >= 0.1) & (freqs <= 0.9))[0]
    if len(idx) == 0:
        return 0.0
    return freqs[idx][np.argmax(fft_mag[idx])] * 60


def augment_window(window):
    noise = np.random.normal(0, 0.005, window.shape)
    scale = random.uniform(0.8, 1.2)
    return (window + noise) * scale


def generate_abnormal_by_bpm(window_data, fs=20):
    orig_bpm = get_dominant_bpm(window_data, fs)
    orig_len = len(window_data)

    target_bpm = random.uniform(25, 45) if random.random() > 0.5 else random.uniform(5, 10)
    ratio = target_bpm / orig_bpm if orig_bpm else 1.0
    new_len = max(int(orig_len / ratio), 1)
    resampled = signal.resample(window_data, new_len)

    if ratio > 1.0:
        output = np.tile(resampled, (int(np.ceil(orig_len / new_len)), 1))[:orig_len, :]
    else:
        output = resampled[:orig_len, :]
        if len(output) < orig_len:
            output = np.pad(output, ((0, orig_len - len(output)), (0, 0)), mode='edge')

    gamma = random.uniform(1.2, 1.5)
    output = np.sign(output) * (np.abs(output) ** gamma)
    output += np.random.normal(0, 0.015, output.shape)
    return augment_window(output)


def is_normal_bpm(bpm, bpm_min=BPM_NORMAL_MIN, bpm_max=BPM_NORMAL_MAX):
    return bpm_min <= bpm <= bpm_max


def label_window(window, fs=TARGET_FREQ, augmentation_count=AUGMENTATION_COUNT):
    """Return labeled (normalized_window, label) samples for one sliding window."""
    window = add_window_noise(window)
    bpm = get_dominant_bpm(window, fs=fs)
    samples = []

    if is_normal_bpm(bpm):
        samples.append((normalize_window(window), 0))
        for _ in range(augmentation_count):
            ab_win = generate_abnormal_by_bpm(window, fs=fs)
            samples.append((normalize_window(ab_win), 1))
    else:
        samples.append((normalize_window(window), 1))

    return samples


def process_single_file(
    file_path,
    fs=TARGET_FREQ,
    window_size=WINDOW_SIZE,
    step_size=STEP_SIZE,
    augmentation_count=AUGMENTATION_COUNT,
):
    processed_data = load_processed_file(file_path, fs=fs, window_size=window_size)
    if processed_data is None:
        return None, None

    x_list, y_list = [], []
    for window in iter_windows(processed_data, window_size, step_size):
        for window_norm, label in label_window(window, fs=fs, augmentation_count=augmentation_count):
            x_list.append(window_norm)
            y_list.append(label)

    if not x_list:
        return None, None
    return np.array(x_list), np.array(y_list)
