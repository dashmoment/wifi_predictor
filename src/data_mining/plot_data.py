import sys
sys.path.append('../')

import pickle
from datetime import datetime
from feature_extraction import feature_engineering
import matplotlib.pyplot as plt
import numpy as np

with open('../../data/raw_data_w_time_sub.pkl', 'rb') as input:

    train = pickle.load(input)
    label_train = pickle.load(input)
    test_raw = pickle.load(input)
    label_test_raw = pickle.load(input)

test_raw = feature_engineering.binding(test_raw)

test_raw.Time = test_raw.Time.apply(lambda x: datetime.strptime(x, '%Y/%m/%d %H:%M:%S:%f'))
test_sorted = test_raw.sort_values(by='Time').reset_index()
label_test_raw['delay_mean'].Time = label_test_raw['delay_mean'].Time.apply(
    lambda x: datetime.strptime(x, '%Y/%m/%d %H:%M:%S:%f'))
label_test_sorted = label_test_raw['delay_mean'].sort_values(by='Time').reset_index()
test_sorted_notime = test_sorted.drop(['index', 'Time'], axis=1)

'''
test_sorted_notime.dropna(axis=0, inplace=True)
label_test_sorted.dropna(axis=0, inplace=True)
'''

#log time series
plt.switch_backend('agg')  # to be able to plot on server

test_sorted_notime_log = test_sorted_notime.apply(lambda x: np.log(x+0.00001))
for column in test_sorted_notime_log:
    plt.figure()
    plt.plot(test_sorted_notime_log[column])
    plt.xlabel('Time')
    plt.title(column)
    plt.savefig('../../analysis_result_rahul/test_sorted_log_{}.png'.format(column))
    plt.close()


