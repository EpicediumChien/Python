import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from collections import defaultdict

# Load and prepare the Iris dataset
iris = load_iris()
X_original = iris.data
y_original = iris.target.astype(float)  # Keep as float for regression

# Create noise pool
noise_pool = np.arange(-0.5, 0.5, 0.01)

# Expand dataset with fuzziness
def expand_with_custom_noise(X, y, times=5):
    X_expanded, y_expanded = [], []
    for _ in range(times):
        noise = np.random.choice(noise_pool, size=X.shape)
        X_noisy = X + noise
        X_expanded.append(X_noisy)
        y_expanded.append(y)
    return np.vstack(X_expanded), np.hstack(y_expanded)

X_big, y_big = expand_with_custom_noise(X_original, y_original, times=5)

# Split
x_train, x_test, y_train, y_test = train_test_split(X_big, y_big, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Model setup
reg = MLPRegressor(
    hidden_layer_sizes=(8, 6, 7),
    activation='logistic',
    solver='adam',
    learning_rate='constant',
    learning_rate_init=0.2,
    max_iter=1,
    warm_start=True,
    random_state=42
)

# Training with tracking
n_epochs = 100
metrics = defaultdict(list)

for epoch in range(n_epochs):
    reg.partial_fit(x_train, y_train)

    y_train_pred = reg.predict(x_train)
    mse = mean_squared_error(y_train, y_train_pred)
    r2 = r2_score(y_train, y_train_pred)

    metrics['epoch'].append(epoch + 1)
    metrics['train_mse'].append(mse)
    metrics['train_r2'].append(r2)

    print(f"Epoch {epoch+1:3d} | MSE: {mse:.4f} | R²: {r2:.4f}")

plt.plot(metrics['epoch'], metrics['train_mse'], label="MSE")
plt.plot(metrics['epoch'], metrics['train_r2'], label="R² Score")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("MLPRegressor Training Metrics")
plt.legend()
plt.grid(True)
plt.show()
