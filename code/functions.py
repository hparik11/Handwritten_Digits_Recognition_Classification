from __future__ import division
from collections import Counter
import random
import numpy as np


def shuffle_values(a, b):
    
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

def entropy(Y):
    
    cnt_y = Counter(Y)
    s = 0.0
    total = len(Y)
    for y, num_y in cnt_y.items():
        probability_y = (num_y/total)
        s += (probability_y)*np.log(probability_y)
    return -s


def entropy_gain(y, y_true, y_false):
    
    return entropy(y) - (entropy(y_true)*len(y_true) + entropy(y_false)*len(y_false))/len(y)


