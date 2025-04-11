import numpy as np
import random
import math
# A user defined function
#===========================Define Factorial=================================
def Factorial(level):
    temp=1
    for i in range(1, level+1):
        temp *= i
    return temp

# number = eval(input("Please input a factorial number: "))
# print(Factorial(number))
#================================Test Power==================================

# number = eval(input("Please input a number: "))
# power = eval(input("Power val: "))
# print(pow(number, power))

#======================================================================
def Exponential(number):
    result, temp = 1, 0
    i = 1
    while i>=1:
        result += number**i/Factorial(i)
        if(round(temp, 14) == round(result, 14)):
            break
        temp = result
        i+=1
    return result

print(Exponential(1))

def exponential_function(x):
    return math.exp(x)

# Example usage
result = exponential_function(1)  # Computes e^2
print(f"e^1 = {result}")