import pandas as pd
import numpy as np
import os

# --- PATHS ---
script_dir = os.path.dirname(os.path.abspath(__file__))
# Check the exact path string again
dyspnea_path = os.path.join(script_dir, 'TrainingData', 'Static Sit', 'imu_log_20251210_223046.csv')

print("--- DIAGNOSTIC REPORT ---")
print(f"Looking for file at:\n{dyspnea_path}\n")

if os.path.exists(dyspnea_path):
    print("✅ File FOUND.")
    
    try:
        df = pd.read_csv(dyspnea_path)
        print(f"✅ CSV Read successfully. Rows: {len(df)}")
        print(f"   Columns found: {list(df.columns)}")
        
        # Check 1: Column Names
        if 'accZ' not in df.columns:
            print("❌ ERROR: Column 'accZ' missing. Check if it's named 'AccZ' or similar.")
        else:
            print("✅ Column 'accZ' exists.")
            
            # Check 2: Sampling Rate & Duration
            if 'timestamp' in df.columns:
                avg_diff = df['timestamp'].diff().mean()
                fs = 1000.0 / avg_diff if avg_diff > 0 else 50.0
                duration = len(df) / fs
                
                print(f"   Calculated FS: {fs:.2f} Hz")
                print(f"   Total Duration: {duration:.2f} seconds")
                
                # Check 3: Logic Requirements
                REQUIRED_OFFSET = 10.0
                REQUIRED_PLOT = 15.0
                TOTAL_NEEDED = REQUIRED_OFFSET + REQUIRED_PLOT
                
                print(f"\n--- LOGIC CHECK ---")
                print(f"Code needs: {TOTAL_NEEDED} seconds ({REQUIRED_OFFSET}s offset + {REQUIRED_PLOT}s plot)")
                
                if duration < TOTAL_NEEDED:
                    print(f"❌ FAILURE: File is too short! ({duration:.2f}s < {TOTAL_NEEDED}s)")
                    print("   -> Solution: Decrease OFFSET_DYSPNEA to 0.0 or record a longer file.")
                else:
                    print("✅ SUCCESS: File is long enough.")
            else:
                print("❌ ERROR: 'timestamp' column missing.")
                
    except Exception as e:
        print(f"❌ CRITICAL ERROR reading CSV: {e}")
else:
    print("❌ FILE NOT FOUND.")
    print("   -> Check if 'TrainingData' or 'Static Sit' folder names have spaces or typos.")
    print(f"   -> Your script is running from: {script_dir}")