import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def compare_raw_magnitude_zoomed(folder_path, file_static, file_fast):
    files = [file_static, file_fast]
    titles = ["Static Stand (Zoomed for Breath)", "Fast Walk (Full Motion)"]
    colors = ['green', 'red']
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

    for idx, file_name in enumerate(files):
        file_path = os.path.join(folder_path, file_name)
        if not os.path.exists(file_path):
            continue
            
        df = pd.read_csv(file_path)
        df.columns = [c.lower().strip() for c in df.columns]
        acc_cols = [c for c in df.columns if 'acc' in c][:3]
        
        # Calculate Raw Magnitude
        raw_values = df[acc_cols].values
        magnitude = np.sqrt(np.sum(raw_values**2, axis=1))
        time_axis = np.arange(len(magnitude)) / 20.0 

        # Plotting
        axes[idx].plot(time_axis, magnitude, color=colors[idx], linewidth=1)
        axes[idx].set_title(f"RAW DATA: {titles[idx]}", fontsize=14)
        axes[idx].set_ylabel("Accel Magnitude")
        axes[idx].grid(True, which='both', alpha=0.3)

        # --- THE FIX: Independent Scaling ---
        if idx == 0: # Static Stand
            # Zooming in around the gravity vector (1.0) 
            # Respiratory signals are usually within 0.02 of the mean
            m_mean = np.mean(magnitude)
            axes[idx].set_ylim(m_mean - 0.05, m_mean + 0.05) 
            axes[idx].annotate('Breathing Micro-waves', xy=(10, m_mean), xytext=(15, m_mean+0.03),
                               arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))
        else: # Fast Walk
            # Leave Fast Walk with a wider view to see the impact spikes
            axes[idx].set_ylim(magnitude.min() - 0.1, magnitude.max() + 0.1)

    axes[1].set_xlabel("Time (seconds)")
    plt.suptitle("Raw IMU Comparison: Highlighting Subtle Respiratory Waves in Static Data", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# --- Execution ---
target_folder = r"C:\Git\Python\IMU\data\Motion Artifact"
compare_raw_magnitude_zoomed(target_folder, "Static Stand.csv", "Fast Walk.csv")