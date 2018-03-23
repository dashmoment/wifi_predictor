#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 10:41:10 2018

@author: ubuntu
"""
import os 

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import scipy.stats as stats
import sklearn.linear_model as linear_model
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import KFold
from IPython.display import HTML, display
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import scipy.stats as st
import math
import mongodb_api as db


d_folder = "../data/ProcessData1070208"
d_file = '1070208small-t2.h5'

d_path = os.path.join(d_folder, d_file)
h5data = pd.HDFStore(d_path)
raw_data= h5data["raw_data"]

data_head5 = pd.DataFrame.from_dict(raw_data['AP']['data'])
label_head5 = pd.DataFrame.from_dict(raw_data['AP']['label'])


        
y = list(label_head5['FER'])
#plt.figure(1); plt.title('FER')
#sns.distplot(y, kde=False, fit=st.lognorm)
#plt.figure(2); plt.title('FER');plt.plot(y)

ping_mean = []
ping_std = []


for i in range(len(label_head5['Ping_mean'])):
    ping_mean.append(np.mean(label_head5['Ping_mean'].loc[i]))
    ping_std.append(np.std(label_head5['Ping_mean'].loc[i]))



features_2_pd = pd.DataFrame()
features_2_pd["FER"] = y
features_2_pd["FER"] = features_2_pd["FER"].apply(pd.to_numeric)
features_2_pd["Ping_mean"] = ping_mean
features_2_pd["Ping_std"] = ping_std

features_2_pd = features_2_pd.fillna(200)

"""
plt.figure(3); plt.title('Ping_mean')
sns.distplot(features_2_pd["Ping_mean"], kde=False, fit=st.lognorm)
plt.figure(4); plt.title('Ping_mean');plt.plot(features_2_pd["Ping_mean"])

plt.figure(5); plt.title('Ping_std')
sns.distplot(features_2_pd["Ping_std"] , kde=False, fit=st.lognorm)
plt.figure(6); plt.title('Ping_std');plt.plot(features_2_pd["Ping_std"] )
"""
data_head5 = pd.DataFrame.from_dict(raw_data['AP']['data'])
label_head5 = pd.DataFrame.from_dict(raw_data['AP']['data'])


for i in range(len(data_head5)):
    for f in data_head5.columns: 
        if len(data_head5[f].loc[i][0]) <= 1:
            data_head5[f].loc[i] = np.float64(data_head5[f].loc[i])
        else:
            data_head5[f].loc[i] = np.reshape(data_head5[f].loc[i],[-1])
             

quantitative = [f for f in data_head5.columns if type(data_head5[f].loc[0]) == np.float64]

for q in quantitative:
    features_2_pd[q] = data_head5[q].apply(pd.to_numeric)



print(features_2_pd['Ping_mean'].head(5))
print(features_2_pd['CRC-ERR'].head(5))



def ping_region_divder(Ping_mean):
    
    if Ping_mean <= 5:  return 0
    elif Ping_mean <= 10 and Ping_mean > 5: return 1
    elif Ping_mean <= 20 and Ping_mean > 10:  return 2
    elif Ping_mean > 20:  return 3

features_2_pd['Ping_mean_cat'] = features_2_pd['Ping_mean']
for i in range(len(features_2_pd)):
    
    features_2_pd['Ping_mean_cat'].loc[i] =  ping_region_divder(features_2_pd['Ping_mean'].loc[i])


SS_protion_keys = []
for i in range(len(data_head5['SS_Portion'].loc[0])):
    SS_protion_keys.append('SS_Portion'+str(i))
    
features_2_pd[SS_protion_keys] = pd.DataFrame(data_head5.SS_Portion.values.tolist(), index= data_head5.index)

SS_Sigval_keys = []
for i in range(len(data_head5['SS_Sigval'].loc[0])):
    SS_Sigval_keys.append('SS_Sigval'+str(i))
features_2_pd[SS_Sigval_keys] = pd.DataFrame(data_head5.SS_Sigval.values.tolist(), index= data_head5.index)

SS_Sigval_std__keys = []
for i in range(len(data_head5['SS_Sigval_Std'].loc[0])):
    SS_Sigval_std__keys.append('SS_Sigval_Std'+str(i))
features_2_pd[SS_Sigval_std__keys] = pd.DataFrame(data_head5.SS_Sigval_Std.values.tolist(), index= data_head5.index)


#Plot pair plot
#sns.pairplot(features_2_pd, x_vars=quantitative + SS_Sigval_keys + SS_protion_keys + ["Ping_mean","Ping_std", "FER"],  y_vars=['Ping_mean','Ping_std', 'FER', 'Ping_mean_cat'])
#Calculate corelation matrix
corr = features_2_pd.corr()

#Plot feature distribution under different labels 
from scipy import stats

guss_model = pd.DataFrame()

for SS_Sigval in range(1):
    
    gmodel = []
    fetature = 'SS_Sigval' + str(SS_Sigval)
    
    for label in range(3,4):
        
        loc_label = np.where(features_2_pd['Ping_mean_cat']==label)
        plt.figure(label); plt.title(fetature +str(label))
        sns.distplot(features_2_pd[fetature].loc[loc_label], kde=False) 
        gmodel.append(stats.norm.fit(features_2_pd[fetature].loc[loc_label]))


    guss_model = pd.concat([guss_model, pd.DataFrame(gmodel, columns=[fetature +'_mu',fetature +'_sigma'])], axis=1)


try:
    corss_model = corss_model.append(guss_model)
except:
    corss_model = guss_model

corss_model = corss_model.reset_index()





