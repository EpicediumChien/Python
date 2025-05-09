import numpy as np

def y(x):
    return (np.sin(2*np.pi*x) + np.sin(5*np.pi*x))

x_vals = np.arange(-1,1,0.002)
y_vals = y(x_vals)

y_max = y_vals.max()
y_vals /= y_max

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_vals, y_vals, test_size=0.20)

import matplotlib.pyplot as plt

plt.figure()
plt.plot(x_train, y_train, 'o')
plt.plot(x_test, y_test, 'x')
plt.show()