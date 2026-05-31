import os
import glob
import random

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from UtilityPy.respiration_preprocess import WINDOW_SIZE, NUM_FEATURES
from UtilityPy.respiration_labels import process_single_file
from TrainingModelPy.respiration_CNNxLSTM import build_model

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print(f"GPU Setup error: {e}")

DATA_DIR = os.path.join('TrainingData', 'Static Sit')


def collect_and_balance(files):
    x_raw, y_raw = [], []
    for f in files:
        xf, yf = process_single_file(f)
        if xf is not None:
            x_raw.append(xf)
            y_raw.append(yf)

    X = np.concatenate(x_raw)
    y = np.concatenate(y_raw)

    idx_0 = np.where(y == 0)[0]
    idx_1 = np.where(y == 1)[0]
    min_samples = min(len(idx_0), len(idx_1))
    np.random.shuffle(idx_0)
    np.random.shuffle(idx_1)

    balanced_idx = np.concatenate([idx_0[:min_samples], idx_1[:min_samples]])
    np.random.shuffle(balanced_idx)
    return X[balanced_idx], y[balanced_idx]


if __name__ == "__main__":
    file_list = glob.glob(os.path.join(DATA_DIR, '*.csv'))
    random.shuffle(file_list)

    print("Processing and Balancing Dataset...")
    split = int(len(file_list) * 0.8)
    X_train, y_train = collect_and_balance(file_list[:split])
    X_test, y_test = collect_and_balance(file_list[split:])

    print(f"Final Training Balance: {np.bincount(y_train.astype(int))}")

    model = build_model((WINDOW_SIZE, NUM_FEATURES))
    history = model.fit(
        X_train, y_train,
        epochs=120,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5),
        ],
    )

    print("\n--- Generating Training Results ---")
    target_names = ['Normal (Eupnea)', 'Abnormal (Dyspnea/Apnea)']

    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.4).astype(int)

    print("\n[Classification Report]")
    print(classification_report(y_test, y_pred, target_names=target_names))

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()

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

    os.makedirs('models', exist_ok=True)
    model.save('models/respiration_CNNxLSTM_classifier.keras')
    print(f"\nTraining Complete. Model saved to 'models/respiration_CNNxLSTM_classifier.keras' (AUC: {roc_auc:.4f})")
