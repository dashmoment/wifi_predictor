# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 17:46:20 2018

@author: RahulJY_Wang
"""
#%%
import os
os.chdir("D:\\RahulJY_Wang\\Qualification\\WiFi_interference_project\\wifi_predictor\\src") #local machine
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import sys
sys.path.append("../")
import matplotlib.pyplot as plt
#plt.switch_backend('agg')
import seaborn as sns
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
from datetime import datetime


from feature_extraction import feature_engineering

from sklearn import metrics
from sklearn import linear_model
from sklearn import svm
from sklearn.svm import SVR
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import make_pipeline
import statsmodels.api as sm
import statsmodels.formula.api as smf
from tensorflow.python.framework import ops
from scipy import stats
from scipy.stats import norm, skew
import xgboost as xgb
from xgboost import plot_importance

from utility import io
import train_test_config as conf
#from feature_extraction import feature_extraction
from feature_extraction import feature_extraction_rahul
from feature_extraction import label_generator as label_gen
from feature_extraction import label_generator_rahul as label_gen_r
from model import ensemble_model 

import importlib
#importlib.reload(main_rahul)
#importlib.reload(feature_extraction_rahul)

#%%
with open('../data/raw_data_w_time_sub.pkl', 'rb') as input:

    train = pickle.load(input)
    label_train = pickle.load(input)
    test = pickle.load(input)
    label_test = pickle.load(input)

### feature engineering: generate features based on SS-subvalue
train = feature_engineering.binding(train)
test = feature_engineering.binding(test)


###sort data
train.Time = train.Time.apply(lambda x: datetime.strptime(x, '%Y/%m/%d %H:%M:%S:%f'))
train_sorted = train.sort_values(by = 'Time').reset_index()
label_train['delay_mean'].Time = label_train['delay_mean'].Time.apply(lambda x: datetime.strptime(x, '%Y/%m/%d %H:%M:%S:%f'))
label_train_sorted = label_train['delay_mean'].sort_values(by = 'Time').reset_index()
train_sorted_notime = train_sorted.drop(['index', 'Time'], axis=1)

train_sorted_notime.dropna(axis=0, inplace=True)
label_train_sorted.dropna(axis=0, inplace=True)

test.Time = test.Time.apply(lambda x: datetime.strptime(x, '%Y/%m/%d %H:%M:%S:%f'))
test_sorted = test.sort_values(by = 'Time').reset_index()
label_test['delay_mean'].Time = label_test['delay_mean'].Time.apply(lambda x: datetime.strptime(x, '%Y/%m/%d %H:%M:%S:%f'))
label_test_sorted = label_test['delay_mean'].sort_values(by = 'Time').reset_index()
test_sorted_notime = test_sorted.drop(['index', 'Time'], axis=1)

test_sorted_notime.dropna(axis=0, inplace=True)
label_test_sorted.dropna(axis=0, inplace=True)


### data exploration
list(train_sorted_notime)
train_sorted_notime.isnull().any()
test_sorted_notime.isnull().any()


### histogram
#pandas histogram (features)
hist = train_sorted_notime.hist(figsize = (50, 30), xlabelsize = 15, ylabelsize = 15)
[x.title.set_size(20) for x in hist.ravel()]
plt.savefig('../analysis_result_rahul/train_sorted_pdhist.png')

hist = train_sorted_notime.apply(lambda x: np.log(x+0.00001)).hist(figsize = (50, 30), xlabelsize = 15, ylabelsize = 15)
[x.title.set_size(20) for x in hist.ravel()]
plt.savefig('../analysis_result_rahul/train_sorted_log_pdhist.png')


#subplot, in one figure (features)
plt.style.use('seaborn-deep')
    
num = 0
plt.figure(figsize=(40,20))
for column in train_sorted_notime:
    num += 1
    plt.subplot(4, 6, num)
    plt.hist([train_sorted_notime[column], test_sorted_notime[column]], label=['train','test'], normed=True)
    plt.legend(loc='upper right')
    plt.title(column)
    plt.ylabel('Frequency')
    plt.xlabel('Value')

plt.show()
plt.savefig('../analysis_result_rahul/train_test_sorted_hist.png')


#labels
plt.figure()
plt.hist([label_train_sorted['Delay-mean'], label_test_sorted['Delay-mean']], bins=20, normed=True)
plt.legend(loc='upper right', labels=['train','test'])
plt.title('Delay-mean')
plt.ylabel('Frequency')
plt.xlabel('Value')



### time series
#plot time series, in one figure (features)
train_sorted.dropna(axis=0, inplace=True)

num = 0
plt.figure(figsize=(40,20))
for column in train_sorted_notime:
    num += 1
    plt.subplot(4, 6, num)
    plt.plot(train_sorted['Time'], train_sorted_notime[column])
    plt.plot(test_sorted['Time'], test_sorted_notime[column])
    plt.legend(loc='upper right', labels=['train', 'test'])
    plt.title(column)
    plt.ylabel('Value')
    plt.xlabel('Time')


#labels
plt.figure()
plt.plot(train_sorted['Time'], label_train_sorted['Delay-mean'])
plt.plot(test_sorted['Time'], label_test_sorted['Delay-mean'])
plt.legend(loc='upper right', labels=['train','test'])
plt.title('Delay-mean')
plt.ylabel('Value')
plt.xlabel('Time')


#plot time series (features)
for column in train_sorted:
    plt.figure()
    plt.plot(train_sorted[column])
    plt.savefig('../analysis_result_rahul/train_sorted_{}.png'.format(column))


#log time series (features)
train_sorted.log = train_sorted.apply(lambda x: np.log(x+0.00001))
for column in train_sorted.apply.log:
    plt.figure()
    plt.plot(train_sorted.log[column])
    plt.savefig('../analysis_result_rahul/train_sorted_log_{}.png'.format(column))


### scatter plot
#transform delay-mean to class
label_train_sorted_class = label_gen_r.transform_delay2category(label_train_sorted['Delay-mean'])
label_test_sorted_class = label_gen_r.transform_delay2category(label_test_sorted['Delay-mean'])

#plot scatter plot
sns.set(style="ticks", color_codes=True)
label_test_sorted_class.name = 'class'
sns.pairplot(pd.concat([test_sorted_notime.iloc[:, :6], label_test_sorted_class], axis=1),
             hue = 'class')

sns.pairplot(pd.concat([test_sorted_notime.iloc[:, 6:12], label_test_sorted_class], axis=1),
             hue = 'class')

sns.pairplot(pd.concat([test_sorted_notime.iloc[:, 12:18], label_test_sorted_class], axis=1),
             hue = 'class')

sns.pairplot(pd.concat([test_sorted_notime.iloc[:, 18:], label_test_sorted_class], axis=1),
             hue = 'class')

sns.pairplot(pd.concat([test_sorted_notime, label_test_sorted_class], axis=1),
             hue = 'class')

### timeit 
start = timeit.default_timer()
test_sorted_notime.apply(np.mean, axis=1)
stop = timeit.default_timer()
print('Time: ', stop - start)

john = test_sorted_notime.values
start = timeit.default_timer()
john.mean(axis=1)
stop = timeit.default_timer()
print('Time: ', stop - start)

np.apply_along_axis(lambda x: sum(-np.abs(x)*np.log(np.abs(x))), axis=1, arr=john)

np.apply_along_axis(np.mean, axis=1, arr=john)

##################
#feature selection
sum(np.argwhere(train_sorted_notime.iloc[:, 0].isnull()) == np.argwhere(label_train_sorted['Delay-mean'].isnull()))
sum(label_train_sorted['Delay-mean'].isnull())



anova_filter = SelectKBest(f_regression, k=5).fit(train_sorted_notime, label_train_sorted['Delay-mean'])
mask = anova_filter.get_support(indices=True)

# ANOVA SVM-C
# 1) anova filter, take 5 best ranked features
anova_filter = SelectKBest(f_regression, k=5)
# 2) svm
clf = SVR(kernel='linear')

anova_svm = make_pipeline(anova_filter, clf)
anova_svm.fit(train_sorted_notime, label_train_sorted['Delay-mean'])
anova_svm.predict(train_sorted_notime).shape

#statsmodel
reg = sm.OLS(label_train['delay_mean'], train)
regfit = reg.fit()
dir(regfit)

regfit.summary2()

#sklearn
reg = linear_model.LinearRegression()
reg.fit(train, label_train['delay_mean'])
reg.coef_
reg.sum

### NN modelling
#Manual Split training and test set
portion = int(len(train)*0.95)

np.random.permutation(5)

# rahul model
X_train = np.array(train.T).reshape(train.shape[1], train.shape[0])
Y_train = np.array(label_train['delay_mean']).reshape(1, label_train['delay_mean'].shape[0])

X_test = np.array(test.T).reshape(test.shape[1], test.shape[0])
Y_test = np.array(label_test['delay_mean']).reshape(1, label_test['delay_mean'].shape[0])

ops.reset_default_graph()
plt.switch_backend('agg')
parameters = main_rahul.model(X_train, Y_train, X_test, Y_test, lr = 0.001, minibatch_size = 256, num_epochs = 500)


# jesse model
train_set = [np.array(train), np.array(label_train['delay_mean'])]
test_set = [np.array(test), np.array(label_test['delay_mean'])]

train_set = [np.array(train), np.array(label_train['delay_mean']).reshape(370319, 1)]
test_set = [np.array(test), np.array(label_test['delay_mean']).reshape(19900,1)]

ops.reset_default_graph()
prediction, loss = main_rahul.general_NN(train_set, test_set, feature = 12, batch_size = 32, Max_epoch=3)

label_test_cat = label_gen.transform_delay2category(pd.DataFrame(label_test['delay_mean']))
pred_test_cat = label_gen.transform_delay2category(pd.DataFrame(prediction))

print('Recall reg_clf:{}'.format(metrics.recall_score(label_test_cat, pred_test_cat, average = 'micro')))
print(metrics.accuracy_score(label_test_cat, pred_test_cat))



###
train_set = [np.array(train_sorted_notime.iloc[:, mask]), 
             np.array(label_train_sorted['Delay-mean'])]
test_set = [np.array(test_sorted_notime.iloc[:, mask]),
            np.array(label_test_sorted['Delay-mean'])]

ops.reset_default_graph()
prediction, loss = main_rahul.general_NN(train_set, test_set, feature = 5, batch_size = 128, Max_epoch=20)

label_test_cat = label_gen.transform_delay2category(pd.DataFrame(label_test_sorted['Delay-mean']))
pred_test_cat = label_gen.transform_delay2category(pd.DataFrame(prediction))

print('Recall reg_clf:{}'.format(metrics.recall_score(label_test_cat, pred_test_cat, average = 'micro')))
print(metrics.accuracy_score(label_test_cat, pred_test_cat))


