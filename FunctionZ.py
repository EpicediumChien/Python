import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split

def z(x,y):
    return np.exp(-(np.square(x) + np.square(y))/0.1)

x = np.arange(-1,1,0.05)
xy = [(j,k) for j in x for k in x]
out = [z(p[0],p[1]) for p in xy]

x_train, x_test, y_train, y_test = train_test_split(xy, out)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# plot train data points
x1_vals = np.array([p[0] for p in x_train])
x2_vals = np.array([p[1] for p in x_train])

ax.scatter(x1_vals, x2_vals, y_train)

# plot test data points
x1_vals = np.array([p[0] for p in x_test])
x2_vals = np.array([p[1] for p in x_test])

ax.scatter(x1_vals, x2_vals, y_test, marker='x')

plt.show()