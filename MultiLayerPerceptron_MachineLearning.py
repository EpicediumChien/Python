import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import mean_squared_error
from collections import defaultdict

# Load and prepare the Iris dataset
iris_dataset = load_iris()
X_original = iris_dataset.data
y_original = iris_dataset.target

# Create a pool of discrete noise values
noise_pool = np.arange(-0.5, 0.5, 0.01)

# Function to expand dataset
def expand_with_custom_noise(X, y, times=5):
    X_expanded = []
    y_expanded = []

    for _ in range(times):
        # Randomly sample noise for each element from the pool
        noise = np.random.choice(noise_pool, size=X.shape)
        X_noisy = X + noise
        X_expanded.append(X_noisy)
        y_expanded.append(y)

    return np.vstack(X_expanded), np.hstack(y_expanded)

X_big, y_big = expand_with_custom_noise(X_original, y_original, times=5)

# Optional: put into DataFrame for inspection
df_big = pd.DataFrame(X_big, columns=iris_dataset.feature_names)
df_big['target'] = y_big

# print(df_big.shape)  # Should be (750, 5)
# print(df_big.head())


x_train, x_test, y_train, y_test = train_test_split(X_big, y_big, test_size=0.2, random_state=42)

# layers=(8,)
# act='tanh'
# sol='adam'
layers=(8,6,7,)
act='relu'
sol='adam'
# Train MLPClassifier
clf = MLPClassifier(
    hidden_layer_sizes=layers,
    activation=act,
    solver=sol,
    learning_rate='constant',
    learning_rate_init=0.2,
    max_iter=1000,
    random_state=42
)
clf.fit(x_train, y_train)

# Prediction and evaluation
y_pred = clf.predict(x_test)
f1 = f1_score(y_test, y_pred, average='weighted')
acc = accuracy_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"F1 Score: {f1:.4f}")
print(f"Accuracy: {acc:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")

# 2D Visualization using PCA
pca = PCA(n_components=2)
x_test_2d = pca.fit_transform(x_test)

# Plot true vs predicted labels
plt.figure(figsize=(10, 5))

# True labels
plt.subplot(1, 2, 1)
plt.title("True Labels")
scatter1 = plt.scatter(x_test_2d[:, 0], x_test_2d[:, 1], c=y_test, cmap='viridis', edgecolors='k')
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.legend(*scatter1.legend_elements(), title="Classes")

# Predicted labels
plt.subplot(1, 2, 2)
plt.title("Predicted Labels")
scatter2 = plt.scatter(x_test_2d[:, 0], x_test_2d[:, 1], c=y_pred, cmap='viridis', edgecolors='k')
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.legend(*scatter2.legend_elements(), title="Classes")

# Create mesh grid in PCA space
h = 0.02  # step size
x_min, x_max = x_test_2d[:, 0].min() - 1, x_test_2d[:, 0].max() + 1
y_min, y_max = x_test_2d[:, 1].min() - 1, x_test_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Flatten grid, then inverse PCA transform to original feature space
grid_pca = np.c_[xx.ravel(), yy.ravel()]
grid_original_space = pca.inverse_transform(grid_pca)

# Predict on the grid
Z = clf.predict(grid_original_space)
Z = Z.reshape(xx.shape)

# Plot decision boundary over the predicted labels plot
plt.subplot(1, 2, 2)
plt.contourf(xx, yy, Z, alpha=0.2, cmap='viridis')

plt.tight_layout()
plt.show()

# # For partial_fit, we need to provide all classes
# classes = np.unique(y_train)

# # Track metrics
# n_epochs = 100
# metrics = defaultdict(list)

# for epoch in range(n_epochs):
#     clf.partial_fit(x_train, y_train, classes=classes)

#     y_train_pred = clf.predict(x_train)
#     acc = accuracy_score(y_train, y_train_pred)
#     f1 = f1_score(y_train, y_train_pred, average='weighted')

#     metrics['epoch'].append(epoch + 1)
#     metrics['train_accuracy'].append(acc)
#     metrics['train_f1'].append(f1)

#     print(f"Epoch {epoch+1:3d} | Accuracy: {acc:.4f} | F1 Score: {f1:.4f}")