import tensorflow as tf
import joblib
import os
import numpy as np

# ==========================================
# Settings
# ==========================================
# keras_path = 'models/respiration_classifier.keras'
keras_path = 'models/movement_classifier.keras'
# scaler_path = 'models/scaler.pkl'
scaler_path = 'models/movement_scaler.pkl'
# output_tflite = 'respiration_model.tflite'
output_tflite = 'movement_model.tflite'

print("--- Step 1: Loading Model ---")
if not os.path.exists(keras_path):
    print(f"❌ Error: Cannot find {keras_path}")
    exit()

model = tf.keras.models.load_model(keras_path)

# --- AUTO-DETECT SHAPES ---
# model.input_shape returns something like (None, 150, 11)
input_shape = model.layers[0].input_shape
if isinstance(input_shape, list): input_shape = input_shape[0]

# Extract the window size and feature count from the model itself
detected_window = input_shape[1]
detected_features = input_shape[2]

print(f"✅ Model Loaded. Detected Input Requirement: Window={detected_window}, Features={detected_features}")

print("\n--- Step 2: Converting to TFLite ---")
try:
    # We use the detected shapes to ensure compatibility
    run_model = tf.function(lambda x: model(x))
    
    concrete_func = run_model.get_concrete_function(
        tf.TensorSpec([1, detected_window, detected_features], model.inputs[0].dtype)
    )

    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

    # Enable support for LSTM/Complex ops
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, 
        tf.lite.OpsSet.SELECT_TF_OPS    
    ]
    converter._experimental_lower_tensor_list_ops = False
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()
    
    with open(output_tflite, 'wb') as f:
        f.write(tflite_model)
    print(f"✅ SUCCESS! TFLite saved as: {output_tflite}")

except Exception as e:
    print(f"❌ Conversion failed: {e}")
    exit()

print("\n--- Step 3: Extracting Scaler for Kotlin ---")
if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
    means = scaler.mean_.tolist()
    stds = scaler.scale_.tolist()

    print(f"private val SCALER_MEANS = floatArrayOf({', '.join([f'{x:.6f}f' for x in means])})")
    print(f"private val SCALER_STDS  = floatArrayOf({', '.join([f'{x:.6f}f' for x in stds])})")
else:
    print("⚠️ Scaler.pkl not found. You will need to extract means/stds manually.")