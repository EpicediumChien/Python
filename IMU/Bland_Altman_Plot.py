import numpy as np
import matplotlib.pyplot as plt

# 1. Ground Truth from Metronome
ground_truth = np.array([10, 14, 18, 22, 26, 30, 34, 38, 42, 46])

# IMU Estimates 
imu_estimate = np.array([6, 12, 12, 12, 24, 24, 30, 30, 36, 36])

# 2. Calculate Mean and Difference
mean_val = (ground_truth + imu_estimate) / 2
diff_val = imu_estimate - ground_truth  

# 3. Statistical Calculations
mean_bias = np.mean(diff_val)
std_diff = np.std(diff_val, ddof=1)
upper_loa = mean_bias + (1.96 * std_diff)
lower_loa = mean_bias - (1.96 * std_diff)

# 4. Plotting
plt.figure(figsize=(12, 7))
plt.scatter(mean_val, diff_val, color='blue', s=100, edgecolors='k', alpha=0.7)

# Add horizontal lines
plt.axhline(mean_bias, color='red', linestyle='-', lw=2, label=f'Mean Bias: {mean_bias:.2f}')
plt.axhline(upper_loa, color='red', linestyle='--', lw=1.5, label=f'+1.96 SD: {upper_loa:.2f}')
plt.axhline(lower_loa, color='red', linestyle='--', lw=1.5, label=f'-1.96 SD: {lower_loa:.2f}')

# Formatting
plt.title('Figure 4: Bland-Altman Plot (IMU vs. Metronome)', fontsize=15, fontweight='bold')
plt.xlabel('Mean of Measurements (BPM)', fontsize=12)
plt.ylabel('Difference (IMU - Metronome) (BPM)', fontsize=12)
plt.legend(loc='upper right', frameon=True)
plt.grid(True, linestyle=':', alpha=0.6)

# --- ADJUSTMENT TO SHOW ALL SPOTS ---
# Calculate the overall min and max to ensure everything is visible
y_min = min(np.min(diff_val), lower_loa)
y_max = max(np.max(diff_val), upper_loa)
# Add a 15% margin so points aren't touching the edge
margin = (y_max - y_min) * 0.15
plt.ylim(y_min - margin, y_max + margin)
# ------------------------------------

plt.tight_layout()
plt.show()

# Print values for verification
print(f"Mean Bias: {mean_bias:.2f} BPM")
print(f"Upper Limit of Agreement (LoA): {upper_loa:.2f} BPM")
print(f"Lower Limit of Agreement (LoA): {lower_loa:.2f} BPM")