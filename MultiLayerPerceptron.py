import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load and prepare the Iris dataset
iris_dataset = load_iris()
X = iris_dataset.data
y = iris_dataset.target
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Train MLPClassifier
clf = MLPClassifier(hidden_layer_sizes=(8,6,4,), activation='logistic', solver='adam', learning_rate='constant', learning_rate_init=0.4, max_iter=1000)
clf.fit(x_train, y_train)

print(accuracy_score(y_test, clf.predict(x_test)))
print(y_train[1:6])
print(clf.coefs_)
print(clf.intercepts_)

# # Get weights from clf.coefs_
# weights_input_to_hidden = clf.coefs_[0]  # shape: (4, 8)
# weights_hidden_to_output = clf.coefs_[1]  # shape: (8, 3)

# # Plot the weights
# fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# # Input to Hidden Layer
# im0 = axs[0].imshow(weights_input_to_hidden, cmap='coolwarm', aspect='auto')
# axs[0].set_title("Weights: Input → Hidden Layer")
# axs[0].set_xlabel("Hidden Neurons")
# axs[0].set_ylabel("Input Features")
# fig.colorbar(im0, ax=axs[0])

# # Hidden to Output Layer
# im1 = axs[1].imshow(weights_hidden_to_output, cmap='coolwarm', aspect='auto')
# axs[1].set_title("Weights: Hidden → Output Layer")
# axs[1].set_xlabel("Output Neurons")
# axs[1].set_ylabel("Hidden Neurons")
# fig.colorbar(im1, ax=axs[1])

# fig.tight_layout()
# plt.show()
