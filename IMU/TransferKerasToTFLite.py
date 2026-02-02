import tensorflow as tf
import joblib
import os
import numpy as np

# ==========================================
# Settings
# ==========================================
keras_path = 'models/breathing_transformer_model.keras'
scaler_path = 'models/scaler.pkl'
output_tflite = 'respiration_model.tflite'

# Must match your training config
WINDOW_SIZE = 150 
FEATURE_COUNT = 11  # (accX,Y,Z, gyroX,Y,Z, roll,pitch,yaw, accMag, gyroMag)

print("--- Step 1: Checking Files ---")
if os.path.exists(keras_path):
    print(f"✅ Found Keras model at {keras_path}")
else:
    print(f"❌ Error: Cannot find {keras_path}")
    exit()

print("\n--- Step 2: Loading Model ---")
try:
    model = tf.keras.models.load_model(keras_path)
    print("✅ Model loaded into memory")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

print("\n--- Step 3: Converting to TFLite (Fixed for LSTM) ---")
try:
    # 1. Convert using a Concrete Function to enforce Static Shapes
    #    We force the batch size to be '1' because on Android you classify one window at a time.
    run_model = tf.function(lambda x: model(x))
    
    # Define the specific input signature: (Batch=1, Time=150, Features=11)
    concrete_func = run_model.get_concrete_function(
        tf.TensorSpec([1, WINDOW_SIZE, FEATURE_COUNT], model.inputs[0].dtype)
    )

    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

    # 2. Enable "Select TF Ops" 
    #    This allows TFLite to fall back to standard TF operations if a specific TFLite op is missing.
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, # Standard TFLite ops
        tf.lite.OpsSet.SELECT_TF_OPS    # Extended TF ops (needed for some LSTM implementations)
    ]

    # 3. Fix for the "TensorListReserve" error
    converter._experimental_lower_tensor_list_ops = False
    
    # 4. Standard Optimization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()
    print("✅ Conversion successful")

except Exception as e:
    print(f"❌ Error during conversion: {e}")
    # Print detailed help if it fails again
    print("\n--- Troubleshooting Tip ---")
    print("If this still fails, your TF version might be too old or too new.")
    print("Try: pip install tensorflow==2.15.0 (or similar stable version)")
    exit()

print("\n--- Step 4: Saving File ---")
try:
    with open(output_tflite, 'wb') as f:
        f.write(tflite_model)
    print(f"✅ SUCCESS! File saved as: {os.path.abspath(output_tflite)}")
except Exception as e:
    print(f"❌ Error saving file: {e}")

print("\n--- Step 5: Extracting Scaler Values ---")
try:
    scaler = joblib.load(scaler_path)
    if hasattr(scaler, 'mean_'):
        print(f"\nIMPORTANT: Copy these to Android ViewModel:")
        # Convert to list for easy copy-pasting
        means = scaler.mean_.tolist()
        scales = scaler.scale_.tolist()
        
        print(f"val MEANS = floatArrayOf({', '.join([f'{x:.6f}f' for x in means])})")
        print(f"val STDS  = floatArrayOf({', '.join([f'{x:.6f}f' for x in scales])})")
    else:
        print("Scaler found but format is unknown.")
except Exception as e:
    print(f"❌ Could not read scaler.pkl: {e}")