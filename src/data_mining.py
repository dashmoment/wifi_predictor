# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 17:46:20 2018

@author: RahulJY_Wang
"""
#%%
import os
os.chdir("/home/rahul/workspace/wifi_predictor/src") #remote machine

import sys
sys.path.append("../")
import matplotlib.pyplot as plt
#plt.switch_backend('agg')
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from scipy import stats
from scipy.stats import norm, skew
from xgboost import plot_importance
from sklearn.metrics import confusion_matrix
import xgboost as xgb

from utility import io
import train_test_config as conf
from feature_extraction import feature_extraction
from feature_extraction import label_generator as label_gen
from model import ensemble_model 
#Manual Split training and test set

#%%
config = conf.train_test_config('Read_Collection_train_c1', 'Read_Collection_test_c1')
fext = feature_extraction.feature_extraction()

train, label_train = fext.generator(config.train, time_step=15)
test, label_test = fext.generator(config.test, time_step=15)

#%%
dir()  #see defined variables
list(train)
train.describe()

hist = train.hist()
plt.savefig('../analysis_result_rahul/train_hist.png')

hist = train.apply(lambda x: np.log(x+0.00001)).hist()
plt.savefig('../analysis_result_rahul/train_hist_log.png')
plt.savefig('../analysis_result_rahul/train_hist_log.png')



#%%
plt.switch_backend('agg') # to be able to run matplotlib on server

#%%
#local machine
os.chdir("D:/RahulJY_Wang/Qualification/WiFi_interference_project/wifi_predictor/src")

#%%
import matplotlib
matplotlib.use('agg') # should run before importing matplotlib.pyplot
import matplotlib.pyplot as plt

