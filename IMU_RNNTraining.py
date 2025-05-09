import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# Read data
def read_and_reshape_imu(filepath, label):
    df = pd.read_csv(filepath, header=None, skiprows=1)
    imu_data = df.iloc[:, 15:24].copy()
    imu_data.loc[:, 'Time'] = df.iloc[:, 0]
    imu_data.loc[:, 'Activity'] = label
    return imu_data

# Load and combine data
sitting = read_and_reshape_imu("C:/Git/R/data/sitting.csv", "Sitting")
standing = read_and_reshape_imu("C:/Git/R/data/standing.csv", "Standing")
walking = read_and_reshape_imu("C:/Git/R/data/walking.csv", "Walking")
all_data = pd.concat([sitting, standing, walking], ignore_index=True)
all_data.columns = ['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'mx', 'my', 'mz', 'Time', 'Activity']

# Shuffle data
all_data = all_data.sample(frac=1, random_state=123).reset_index(drop=True)

# Parameters
# 3 sec per group
group_size = 300
num_features = 9

# Calculate number of groups
num_groups = all_data.shape[0] // group_size

# Prepare x_array
x_array = np.zeros((num_groups, group_size, num_features))
for i in range(num_groups):
    start = i * group_size
    end = start + group_size
    x_array[i] = all_data.iloc[start:end, 0:9].astype(float).values

# Prepare y_array (majority label per group)
# activity_map = {'Sitting': 0, 'Standing': 1, 'Walking': 2}
# activity_map = {'Standing': 0, 'Sitting': 1, 'Walking': 2}
# activity_map = {'Standing': 0, 'Walking': 1, 'Sitting': 2}
activity_map = {'Walking': 0, 'Sitting': 1, 'Standing': 2}

y_labels = []
for i in range(num_groups):
    start = i * group_size
    end = start + group_size
    group_labels = all_data['Activity'].iloc[start:end]
    majority_label = group_labels.mode()[0]
    y_labels.append(majority_label)
y_labels_int = np.array([activity_map[label] for label in y_labels])
y_array = to_categorical(y_labels_int, num_classes=3)

# Build RNN model
model = Sequential()
model.add(SimpleRNN(units=32, input_shape=(group_size, num_features)))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
history = model.fit(x_array, y_array, epochs=20, batch_size=16, validation_split=0.2)

# Load and prepare test data
test_data = pd.read_csv("C:/Git/R/homework.csv", header=None, skiprows=1)
test_data.columns = ['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'mx', 'my', 'mz', 'Activity']
test_data['Activity'] = test_data['Activity'].astype(int)
num_test_groups = test_data.shape[0] // group_size

test_x = np.zeros((num_test_groups, group_size, num_features))
for i in range(num_test_groups):
    start = i * group_size
    end = start + group_size
    test_x[i] = test_data.iloc[start:end, 0:9].astype(float).values

# Prepare test labels (majority per group)
test_y_labels = []
for i in range(num_test_groups):
    start = i * group_size
    end = start + group_size
    group_labels = test_data['Activity'].iloc[start:end]
    majority_label = group_labels.mode()[0]
    test_y_labels.append(majority_label)
test_y_labels_int = np.array(test_y_labels, dtype=int)

# Predict
predictions = model.predict(test_x)
predicted_labels = np.argmax(predictions, axis=1)

# Report
print(classification_report(test_y_labels_int, predicted_labels))
cm = confusion_matrix(test_y_labels_int, predicted_labels)
print("Confusion Matrix:\n", cm)

# Save model
model.save("recurrent_neural_network.h5")

# Calculate and print macro F1 score
macro_f1 = f1_score(test_y_labels_int, predicted_labels, average='macro')
print(f"Macro-average F1 score: {macro_f1:.4f}")

# Save model
model.save("recurrent_neural_network.h5")