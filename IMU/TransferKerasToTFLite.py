import tensorflow as tf
import numpy as np
import os

# ==========================================
# 1. 設定路徑
# ==========================================
keras_path = 'models/respiration_CNNxLSTM_classifier.keras'
output_tflite = 'respiration_CNNxLSTM_classifier.tflite'

print("--- Step 1: Loading Model ---")
if not os.path.exists(keras_path):
    print(f"❌ Error: {keras_path} not found")
    exit()

# 載入原始模型 (不使用 safe_mode 以增加相容性)
model = tf.keras.models.load_model(keras_path, safe_mode=False)
print("✅ Model Loaded.")

# ==========================================
# 2. 重新包裝模型 (解決 Untracked Resource 與 Version 12 問題)
# ==========================================
print("\n--- Step 2: Re-wrapping Model with Fixed Shape [1, 200, 11] ---")

try:
    # 建立一個全新的輸入層，強制 Batch Size = 1
    # 這能確保轉出的 FULLY_CONNECTED 是手機支援的舊版本 (v9/v10)
    input_layer = tf.keras.layers.Input(batch_shape=(1, 200, 11), name='input_fixed')
    
    # 將資料餵入舊模型，強制設定 training=False 排除 Dropout 的干擾
    output_layer = model(input_layer, training=False)
    
    # 建立一個新的 Functional Model
    fixed_model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
    print("✅ Fixed Model Graph Re-built.")

# ==========================================
# 3. 執行轉換
# ==========================================
    print("\n--- Step 3: Converting to TFLite ---")
    
    converter = tf.lite.TFLiteConverter.from_keras_model(fixed_model)

    # 針對 LSTM 的核心設定
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, 
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    
    # 關閉優化 (這在 Keras 3 下最安全，避免產生 v12 運算元)
    converter.optimizations = []
    
    # 執行轉換
    tflite_model = converter.convert()
    
    with open(output_tflite, 'wb') as f:
        f.write(tflite_model)
    
    print(f"\n✅ 最終成功！產出的檔案版本已降級至相容版。")
    print(f"請將此檔案放入手機 assets: {output_tflite}")

except Exception as e:
    print(f"❌ 轉換失敗具體原因: {e}")