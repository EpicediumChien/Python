import random
from collections import Counter
import math
def experiment(): # simulates a single trial of the experiment
    data = ['blue'] * 5 + ['red'] * 3 + ['white'] * 3  
    draw = random.sample(data, 3)      
    return Counter(draw)

def compute_probability(num_trials=10000):
    """Repeat the experiment and compute the probability."""
    case_a = 0  # One blue, one red, one white
    case_b = 0  # All three are red
    case_c = 0  # Exactly two blue balls

    for _ in range(num_trials):
        outcome = experiment()
        
        # Check case (a): one blue, one red, one white
        if outcome['blue'] == 1 and outcome['red'] == 1 and outcome['white'] == 1:
            case_a += 1
        
        # Check case (b): all three are red
        if outcome['red'] == 3:
            case_b += 1
        
        # Check case (c): exactly two blue balls
        if outcome['blue'] == 2:
            case_c += 1

    prob_a = case_a / num_trials
    prob_b = case_b / num_trials
    prob_c = case_c / num_trials

    return prob_a, prob_b, prob_c
    
# Main program
num_num_trials = 100000  
prob_a, prob_b, prob_c = compute_probability(num_num_trials)

print(f"Probability of case a: {prob_a:.4f}")
print(f"Probability of case b: {prob_b:.4f}")
print(f"Probability of case c: {prob_c:.4f}")

print(f"Combination of 5,3 : {math.comb(5,3)}")
print(f"Permutation of 5,3 : {math.perm(5,3)}")

"""
def is_vowel(letter):
    return letter in 'aeiou'

import random
def experiment(): # simulates a single trial of the experiment
    alphabet= list("abcdefghijklmnopqrstuvwxyz")  
    draw = random.sample(alphabet, 2)  
    letter1, letter2 = draw[0], draw[1] 
    return (is_vowel(letter1) and not is_vowel(letter2)) or (not is_vowel(letter1) and is_vowel(letter2))

def compute_probability(num_trials=10000):
    #Repeat the experiment and compute the probability.
    success_count = 0
    for _ in range(num_trials):
        if experiment():
            success_count += 1    
    return success_count / num_trials

# Main program
num_trials = 10000
probability = compute_probability(num_trials)
print(f"Estimated probability: {probability:.4f}")
"""