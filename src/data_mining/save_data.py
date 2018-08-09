import sys
sys.path.append("../")

import matplotlib.pyplot as plt
# plt.switch_backend('agg')
import numpy as np
import pandas as pd

from utility import io
import train_test_config as conf
from feature_extraction import feature_extraction
from feature_extraction import feature_extraction_rahul
from feature_extraction import label_generator as label_gen

import pickle
import importlib
# importlib.reload(feature_extraction_rahul)

# feature extraction with time
config = conf.train_test_config('Read_Collection_train_c1', 'Read_Collection_test_c1')
fext = feature_extraction_rahul.feature_extraction()

train, label_train = fext.generator(config.train, time_step=15, special_list='SS_Subval')
test, label_test = fext.generator(config.test, time_step=15, special_list='SS_Subval')

with open('../../data/raw_data_w_time_sub.pkl', 'wb') as output:
    pickle.dump(train, output, pickle.HIGHEST_PROTOCOL)
    pickle.dump(label_train, output, pickle.HIGHEST_PROTOCOL)
    pickle.dump(test, output, pickle.HIGHEST_PROTOCOL)
    pickle.dump(label_test, output, pickle.HIGHEST_PROTOCOL)


# feature extraction without time
fext = feature_extraction.feature_extraction()

train, label_train = fext.generator(config.train, time_step=15, special_list='SS_Subval')
test, label_test = fext.generator(config.test, time_step=15, special_list='SS_Subval')

with open('../../data/raw_data_wo_time_sub.pkl', 'wb') as output:
    pickle.dump(train, output, pickle.HIGHEST_PROTOCOL)
    pickle.dump(label_train, output, pickle.HIGHEST_PROTOCOL)
    pickle.dump(test, output, pickle.HIGHEST_PROTOCOL)
    pickle.dump(label_test, output, pickle.HIGHEST_PROTOCOL)




'''
import shelve

filename = '../data/raw_data_shelve.out'
my_shelf = shelve.open(filename, 'n')

for key in dir():
    try:
        my_shelf[key] = globals()[key]
    except TypeError:
        print('ERROR shelving : {0}'.format(key))

my_shelf.close()
'''
