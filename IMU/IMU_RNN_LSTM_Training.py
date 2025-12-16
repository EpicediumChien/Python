import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

activity_map = {'Walking': 0, 'Sitting': 1, 'Standing': 2}

def load_and_window(filepath, label, window_seconds=0.5):
    df = pd.read_csv(filepath)
    df = df[['time', 'ax', 'ay', 'az', 'mx', 'my', 'mz', 'gx', 'gy', 'gz']]
    
    # Calculate sampling rate
    time_diff = df['time'].iloc[1] - df['time'].iloc[0]
    sampling_rate = int(1.0 / time_diff)
    group_size = int(window_seconds * sampling_rate)
    print(f"{label}: Sampling Rate = {sampling_rate} Hz, Group Size = {group_size}")
    
    num_windows = df.shape[0] // group_size
    x = np.zeros((num_windows, group_size, 9))  # 9 features
    y = np.full((num_windows,), activity_map[label])
    
    for i in range(num_windows):
        start = i * group_size
        end = start + group_size
        x[i] = df.iloc[start:end, 1:].values.astype(float)
    
    return x, y

# Load and window each activity separately
x_sit, y_sit = load_and_window("C:/Git/Python/IMU/data/sitting.csv", "Sitting")
x_stand, y_stand = load_and_window("C:/Git/Python/IMU/data/standing.csv", "Standing")
x_walk, y_walk = load_and_window("C:/Git/Python/IMU/data/walking.csv", "Walking")

# Combine all windows
x_all = np.concatenate([x_sit, x_stand, x_walk], axis=0)
y_all = np.concatenate([y_sit, y_stand, y_walk], axis=0)
y_cat = to_categorical(y_all, num_classes=3)

# Stratified train-test split
x_train, x_test, y_train, y_test, y_train_int, y_test_int = train_test_split(
    x_all, y_cat, y_all, test_size=0.3, stratify=y_all, random_state=42
)

# Print label distributions
print("Train label counts:", np.bincount(y_train_int))
print("Test label counts :", np.bincount(y_test_int))

# Build and train model
model = Sequential()
model.add(LSTM(32, input_shape=(x_all.shape[1], x_all.shape[2])))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=20, batch_size=16, validation_data=(x_test, y_test))

# Predict
predictions = model.predict(x_test)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = np.argmax(y_test, axis=1)

# Evaluation
print("\nClassification Report:")
print(classification_report(true_labels, predicted_labels, target_names=["Walking", "Sitting", "Standing"]))
print("Confusion Matrix:")
print(confusion_matrix(true_labels, predicted_labels))
print(f"Macro F1 Score: {f1_score(true_labels, predicted_labels, average='macro'):.4f}")

# Compute mean of x, y, z axes across window for each sample
# Shape: (num_samples, 9)
means = np.mean(x_all, axis=1)

# Group sensor axis by spatial axes
x_param = (means[:, 0] + means[:, 3] + means[:, 6]) / 3  # ax, mx, gx
y_param = (means[:, 1] + means[:, 4] + means[:, 7]) / 3  # ay, my, gy
z_param = (means[:, 2] + means[:, 5] + means[:, 8]) / 3  # az, mz, gz

# Prepare 3D plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

colors = ['r', 'g', 'b']
labels = ['Walking', 'Sitting', 'Standing']

for activity_id in range(3):
    idx = y_all == activity_id
    ax.scatter(
        x_param[idx], y_param[idx], z_param[idx],
        c=colors[activity_id], label=labels[activity_id], s=15, alpha=0.7
    )

ax.set_title("3D IMU Feature Space: Aggregated X, Y, Z Axes")
ax.set_xlabel("X-axis (ax, gx, mx avg)")
ax.set_ylabel("Y-axis (ay, gy, my avg)")
ax.set_zlabel("Z-axis (az, gz, mz avg)")
ax.legend()
plt.show()