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

class Solution:
    def maxArea(self, height: List[int]) -> int:
        head = 0
        tail = len(height) - 1
        max_result = 0

        while head < tail:
            # Calculate the current area
            current_height = min(height[head], height[tail])
            current_width = tail - head
            current_area = current_height * current_width

            # Update max_result if the current area is larger
            max_result = max(max_result, current_area)

            # Move the pointer of the shorter line inwards
            if height[head] < height[tail]:
                head += 1
            else:
                tail -= 1

        return max_result