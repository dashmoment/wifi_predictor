import mongodb_api as db
import pandas as pd
import sys
import numpy as np
from sklearn.svm import SVR
import random
import os
from matplotlib import pyplot as plt
import copy

import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew #for some statistics

color = sns.color_palette()
sns.set_style('darkgrid')


db_collections =  {'collection':['1070222-clear', '1070223-one20M', '1070223-one10M', 
                        '1070227-two10M', '1070227-two10Mt2','1070227-two10Mt2' 
                        '1070301-two20Mt2', '1070301-two20M', '1070302-two20M',
                        '1070302-two20M-L3is30'],
                 'number':[5,3,3,3,3,3,3,3,3]
                 }
                      
                  
device = 'AP'
AP_data = []
                 
for i in range(len(db_collections['collection'])):

    collection_name = db_collections['collection'][i]
    
    for j in range(db_collections['number'][i]):
        
        
        if j==0:
            db_collection = db_collections['collection'][i] + '-ProcessData'
        else:
            db_collection = db_collections['collection'][i] + '-'+ str(j+1) + '-ProcessData'
        
        mW = db.mongodb_api(user='ubuntu', pwd='ubuntu', database='wifi_diagnosis',collection=db_collection)
        fdata = mW.find(key_value = {}, ftype='many')


        for k in range(len(fdata)):
            AP_data.append(pd.Series(fdata[k]['AP']))
            #tmp = pd.Series.transpose(tmp)
    
AP_data = pd.concat(AP_data, axis=1).transpose()


#Drop data with no delay values 
AP_data = AP_data[AP_data.astype(str)['Delay']!='[]'].reset_index(drop='True')
AP_data = AP_data[AP_data.astype(str)['SS_Subval']!='[]'].reset_index(drop='True')


delay_raw = []
delay_raw_mean = []
for i in range(AP_data.shape[0]):
    delay_raw = delay_raw + AP_data['Delay'].values[i]
    delay_raw_mean.append(np.mean(AP_data['Delay'].values[i]))


delay_raw = np.array(delay_raw)
delay_raw_mean = np.array(delay_raw_mean)

plt.figure()
sns.distplot(delay_raw[delay_raw < 100], bins=[0,5,10,15,20,100], fit=norm)
bins_number = np.histogram(delay_raw[delay_raw < 100], bins=[0,5,10000])

plt.figure()
sns.distplot(delay_raw_mean[delay_raw_mean < 100],  bins=[0,5,10,15,20,100], fit=norm)

delay_raw_log = np.log1p(delay_raw)
plt.figure()
sns.distplot(delay_raw_log[delay_raw_log < 100], fit=norm)


AP_data['Delay_mean'] = delay_raw_mean

delay = pd.DataFrame(AP_data['Delay_mean'])
c = pd.cut(
       delay.stack(),
       [0,5, np.inf],
       labels = [0,1]
       )
delay = delay.join(c.unstack().add_suffix('_cat'))

sigvalue_mean = []
for i in range(AP_data.shape[0]):
    sigvalue_mean.append( np.mean(AP_data['SS_Subval'][i], axis=0))
    
sig_chan = ['sig_ch_'+str(i) for i in range(56)]

delay[sig_chan] = pd.DataFrame(sigvalue_mean)
corrmat = delay.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True)

train = delay[:6000]
test = delay[6000:]

#Check label distribution
train_bins_number = np.histogram(train['Delay_mean_cat'], bins=[0,1,2])
test_bins_number = np.histogram(test['Delay_mean_cat'], bins=[0,1,2])

print('Train label distribution:',train_bins_number)
print('Test label distribution:',test_bins_number)


import xgboost as xgb
from xgboost import plot_tree
from xgboost import plot_importance
from sklearn.metrics import confusion_matrix

#train_label_mask = np.where(np.array(train_label_oh) == 0, 0 ,1)

model = xgb.XGBClassifier(max_depth=5, learning_rate=0.01 ,n_estimators=2000, silent=False)
model.fit(train[sig_chan], train['Delay_mean_cat'])

#plot_tree(model)
#plt.show()

xg_prediction_train = model.predict(train[sig_chan])
xg_accuracy_train = np.mean(np.equal(np.array(train['Delay_mean_cat']), xg_prediction_train).astype(np.float32))
xg_train_c_matrix = confusion_matrix(np.array(train['Delay_mean_cat']), xg_prediction_train)
plt.imshow(xg_train_c_matrix)
plt.show()

xg_prediction_test = model.predict(test[sig_chan])
xg_accuracy_test = np.mean(np.equal(np.array(test['Delay_mean_cat']), xg_prediction_test).astype(np.float32))
xg_test_c_matrix = confusion_matrix(np.array(test['Delay_mean_cat']), xg_prediction_test)
plt.imshow(xg_test_c_matrix)
plt.show()


plot_importance(model)
plt.show()


#Drop data with delay < 5
delay_drop_c1 = delay[delay['Delay_mean'] > 5].reset_index(drop=True)
sns.distplot(delay_drop_c1['Delay_mean'], bins=[5,10,15,20])
delay_drop_c1 = delay_drop_c1.drop('Delay_mean_cat',1)

c = pd.cut(
       pd.DataFrame(delay_drop_c1['Delay_mean']).stack(),
       [5,10, np.inf],
       labels = [0,1]
       )
delay_drop_c1 = delay_drop_c1.join(c.unstack().add_suffix('_cat'))
print(np.histogram(delay_drop_c1['Delay_mean_cat'], bins=[0,1,2]))

train = delay_drop_c1[:3000]
test = delay_drop_c1[3000:]
train_bins_number = np.histogram(train['Delay_mean_cat'], bins=[0,1,2])
test_bins_number = np.histogram(test['Delay_mean_cat'], bins=[0,1,2])

print('Train label distribution:',train_bins_number)
print('Test label distribution:',test_bins_number)

model.fit(train[sig_chan], train['Delay_mean_cat'])
xg_prediction_train = model.predict(train[sig_chan])
xg_accuracy_train = np.mean(np.equal(np.array(train['Delay_mean_cat']), xg_prediction_train).astype(np.float32))
xg_train_c_matrix = confusion_matrix(np.array(train['Delay_mean_cat']), xg_prediction_train)
plt.imshow(xg_train_c_matrix)
plt.show()

xg_prediction_test = model.predict(test[sig_chan])
xg_accuracy_test = np.mean(np.equal(np.array(test['Delay_mean_cat']), xg_prediction_test).astype(np.float32))
xg_test_c_matrix = confusion_matrix(np.array(test['Delay_mean_cat']), xg_prediction_test)
plt.imshow(xg_test_c_matrix)
plt.show()

