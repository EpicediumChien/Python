import random
import numpy as np
import pandas as pd
# import torch

# data1 = [[1, 5, 8, 7, 6], [5, 8, 7, 6, 4]]
# array1 = np.array(data1)
# arrange1 = np.arange(start=1,stop=15, step=3, dtype=float)
# random1 = np.random.randint(low=1, high=10, size=(5,5))
# random2 = np.random.rand(5, 5)
# random2 = random2*100
# lineSpace = np.linspace(0, 10, 10)

# print(array1)
# print(arrange1)
# print(random1)
# print(random2)
# print(lineSpace)
# print("Shape: " + str(array1.shape))
# print("Ndim: " + str(array1.ndim))
# print("ItemSize: " + str(array1.itemsize)) 

#=================================Mean and Std=====================================

# data2 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# array2 = np.array(data2)
# array2 = array2.reshape(3,3)
# print(array2)

# for row in range(array2.ndim+1):
#     print("Avg:", np.mean(array2[row]),\
#         " Std: ", np.std(array2[row]),\
#         "\n")
    
#=====================================Star=========================================
n = int(input())
cap = 2*n-1
arr = list(range(cap,-1,-2))
for i in arr:
    print("*"*i + "\n")

print(arr)
# print(11//3)

# x=20
# def func(x):
#     x=10
# func(x)
# print("x is", x)

#=====================================Numpy=========================================