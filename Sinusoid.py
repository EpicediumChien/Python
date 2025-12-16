import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

def y(x):
    return (np.sin(2*np.pi*x) + np.sin(5*np.pi*x))

x_vals = np.arange(-1,1,0.002)
y_vals = y(x_vals)

y_max = y_vals.max()
y_vals /= y_max

x_train, x_test, y_train, y_test = train_test_split(x_vals, y_vals, test_size=0.20)

plt.figure()
plt.plot(x_train, y_train, 'o')
plt.plot(x_test, y_test, 'x')
# plt.show()

x_train = x_train.reshape(-1,1)
x_test = x_test.reshape(-1,1)

mlp = MLPRegressor(
    # Test different layers of nodes
    # hidden_layer_sizes=[30, 30, 50, 50, 80, 20],
    hidden_layer_sizes=[30, 30],
    activation='relu',
    # Test different activation function
    # activation='tanh',
    solver='lbfgs',
    # solver='adam',
    max_iter=1000,
)

mlp.fit(x_train,y_train)

predictions = mlp.predict(x_test)
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse:.6f}")

plt.plot(x_test, predictions, 'ro')
plt.show()