import numpy as np
import matplotlib.pyplot as plt

# 1. Ground Truth from Metronome
ground_truth = np.array([10, 14, 18, 22, 26, 30, 34, 38, 42, 46])

# IMU Estimates from your specific plots
imu_estimate = np.array([13.99, 17.02, 17.99, 19.99, 25.03, 25.99, 31.98, 34.98, 38.05, 45.99])

# 2. Calculate Mean and Difference
mean_val = (ground_truth + imu_estimate) / 2
diff_val = imu_estimate - ground_truth  # Estimate - Reference

# 3. Statistical Calculations
mean_bias = np.mean(diff_val)
std_diff = np.std(diff_val, ddof=1)
upper_loa = mean_bias + (1.96 * std_diff)
lower_loa = mean_bias - (1.96 * std_diff)

# 4. Plotting
plt.figure(figsize=(10, 6))
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

# Set limits for clarity
plt.ylim(mean_bias - 10, mean_bias + 10)

plt.tight_layout()
plt.show()

# Print values for your paper text
print(f"Mean Bias: {mean_bias:.2f} BPM")
print(f"Upper Limit of Agreement (LoA): {upper_loa:.2f} BPM")
print(f"Lower Limit of Agreement (LoA): {lower_loa:.2f} BPM")