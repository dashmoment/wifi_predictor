import matplotlib.pyplot as plt
# plt.switch_backend('agg')
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn import linear_model
from sklearn import svm
from sklearn.svm import SVR
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import make_pipeline

from tensorflow.python.framework import ops
from scipy import stats
from scipy.stats import norm, skew
import xgboost as xgb
from xgboost import plot_importance
import keras
import keras.backend as K

from feature_extraction import label_generator as label_gen
from feature_extraction import label_generator_rahul as label_gen_r
from feature_extraction import feature_engineering
from feature_extraction import feature_extraction_sample_ma
from model.nn_model_rahul_keras import nn_model
from utility import io
import train_test_config as conf

if __name__ == '__main__':
    
    #Load train and test configuration
    config = conf.train_test_config('Read_Collection_train_c1', 'Read_Collection_test_ct')
        
    #Generator train & test data by configuration 
    fext = feature_extraction_sample_ma.feature_extraction()
    train, label_train, test, label_test = fext.generator(config.train, time_step=15,special_list = ['SS_Subval'])
    
    train = feature_engineering.binding(train)
    test = feature_engineering.binding(test)
    
    if 'Time' in train.columns:
        train.drop('Time',axis=1, inplace=True)
        test.drop('Time',axis=1, inplace=True)
        
        
    anova_filter = SelectKBest(f_regression, k=12).fit(train, label_train['delay_mean'])
    mask = anova_filter.get_support(indices=True)  
    print('Selected features from regression: {}'.format(train.columns[mask]))
    
    
    train = train.sample(frac=1.0)    
    label_train_shuffle = label_train['delay_mean'].iloc[train.index]
    test = test.sample(frac=1.0)    
    label_test_shuffle = label_test['delay_mean'].iloc[train.index]
    
    
data_length = 300000
time_step = 15
num_interval = 4000
# === a function to sample training and testing interval with data length given
maxInt = data_length//time_step - 1
selectInt = np.random.choice(np.arange(maxInt), num_interval, replace=False)
selectInt = np.sort(selectInt)

testing_interval = [x for x in zip(selectInt*time_step, (selectInt+1)*time_step)]
if selectInt[0] == 0: # deal with special case when testing interval starting with 0
    training_interval = [x for x in zip(selectInt[1:]*time_step, (selectInt[:-1]+1)*time_step)]
else:
    training_interval = [x for x in zip(np.append(0, selectInt[:-1]+1)*time_step, selectInt*time_step)]
training_interval.append([training_interval[-1][1], data_length])
training_interval = [x for x in training_interval if x[1]-x[0] != 0]
