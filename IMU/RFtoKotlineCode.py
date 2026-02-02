import joblib
import m2cgen as m2c
import os

# 1. Load your stable Random Forest model
model_path = 'models/breathing_rf_stable.pkl'
if not os.path.exists(model_path):
    print(f"Error: {model_path} not found!")
else:
    model = joblib.load(model_path)

    try:
        # Try exporting to Kotlin first
        print("Attempting to export to Kotlin...")
        code = m2c.export_to_kotlin(model)
        filename = "Model.kt"
    except AttributeError:
        # Fallback to Java if Kotlin exporter is missing
        print("Kotlin exporter not found. Exporting to Java instead (Compatible with Kotlin)...")
        code = m2c.export_to_java(model)
        filename = "Model.java"

    # 2. Save the code
    with open(filename, "w") as f:
        f.write(code)
    
    print(f"Successfully generated {filename}!")