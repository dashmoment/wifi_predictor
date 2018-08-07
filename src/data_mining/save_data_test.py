import sys
sys.path.append("../")

import numpy as np
import pandas as pd
import pickle

with open('../data/raw_data.pkl', 'rb') as input:
    train = pickle.load(input)
    label_train = pickle.load(input)
    test = pickle.load(input)
    label_test = pickle.load(input)

print(train.head(3))
print(label_train['delay_mean'].head(3))

'''
import shelve

filename = '../data/raw_data_shelve.out'
my_shelf = shelve.open(filename)
for key in my_shelf:
    globals()[key] = my_shelf[key]
my_shelf.close()

print(dir())
'''
