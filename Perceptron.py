# import numpy as np

# a = np.array([[1, 0.85, 0.05]])
# b = np.array([[2.73, 3.51],
#         [-6.37, 7.89],
#         [6.60, -7.92]])
# print(1/(1+np.exp(-np.matmul(a, b))))

# Lecture I
# import numpy as np
# inputs = []
# inputs.append(np.array([1, 1, 0]))
# inputs.append(np.array([1, 0, 1]))
# inputs.append(np.array([0, 1, 1]))
# inputs.append(np.array([0, 0, 0]))
# labels = np.array([1, 0, 0, 0])
# Iters = 25
# # adjust to the min acceptable Iters
# no_of_inputs = 2
# weights = np.random.randn(no_of_inputs + 1) # [-5, -5, -5]
# # because -5, -5, -5 is too far from the real value
# print("initial: " + str(weights))
# learning_rate = 0.15
# error = []
# for _ in range(Iters):
#     count = 0
#     for _input, label in zip(inputs, labels):
#         summation = np.dot(_input, weights) # dot
#         if summation > 0:
#             predicted = 1
#         else:
#             predicted = 0
#         weights += learning_rate * (label - predicted) * _input
#         if abs(label - predicted) == 1:
#             count += 1
#     error.append(count)
# print("trained: " + str(weights))
# import matplotlib.pyplot as plt
# plt.plot(error)
# plt.show()
    



# Lecture II
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
iris = datasets.load_iris()
#features = iris.data
features = iris.data[:,2:4]
target = iris.target
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2)

#sc = StandardScaler()
#sc.fit(features_train)
#features_train_std = sc.transform(features_train) # feature scaling
#features_test_std = sc.transform(features_test)

clf = Perceptron(max_iter=100, eta0=0.15, random_state=None)
#clf.fit(features_train_std, target_train)
clf.fit(features_train, target_train)

#target_pred = clf.predict(features_test_std)
target_pred = clf.predict(features_test)
print(accuracy_score(target_test, target_pred))
print(clf.coef_)
print(clf.intercept_)