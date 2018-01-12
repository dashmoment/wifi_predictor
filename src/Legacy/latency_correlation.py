#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 14:31:48 2017

@author: ubuntu
"""

import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import os
import random
from scipy.stats.stats import pearsonr  
#import model_zoo as mz

#import tensorflow.contrib.eager as tfe
from matplotlib import pyplot as plt

h5data = pd.HDFStore('../data/scan_data_1061129.h5')
raw_data= h5data["raw_data"]

h5data2 = pd.HDFStore('../data/scan_data_1061205.h5')
raw_data2= h5data2["raw_data"]



data_type = "Portion"
label_type = "Spectralscan_mean"

data_1129 = []
label_1129 = []
label_dict = {}
for i in range(len(raw_data)):
    label_dict[raw_data[label_type][i]] = {}
    label_dict[raw_data[label_type][i]]["1129"] = i
    data_1129.append(raw_data[data_type][i])
    label_1129.append(raw_data[label_type][i])


data_1205 = []
label_1205 = []
for i in range(len(raw_data2)):
    
    data_1205.append(raw_data2[data_type][i])
    label_1205.append(raw_data2[label_type][i])
    
    for key in label_dict:
        if abs(raw_data2[label_type][i] - key) < 1:
            
            try:
                label_dict[key]["1205"].append(i)
            except:
                label_dict[key]["1205"] = [i]
            break
        



data_1129 = np.stack(data_1129)
label_1129 = np.stack(label_1129)
data_1205 = np.stack(data_1205)
label_1205 = np.stack(label_1205)

correlate_dict = {}

process = 0
for k in label_dict:
    
    print("Process: {}/{}".format(process, len(label_dict)))
    process += 1
    
    try:
        
        value_1129 = label_dict[k]["1129"]
        value_1205 = label_dict[k]["1205"]
        
        correlate_dict[label_1129[value_1129]] = []
        
        for i in range(len(value_1205)):
            
            cof, pval = pearsonr(data_1129[value_1129,:],data_1205[value_1205[i],:]) 
            correlate_dict[label_1129[value_1129]].append([cof, pval, value_1129, value_1205[i]])
        
    except:
        continue





mean_cof = []
for k in correlate_dict:
    
    tmp = np.stack(correlate_dict[k])[:,0]
    tmp = [x for x in tmp if str(x) != 'nan']
    
    if len(tmp) > 0 : mean_cof.append([k, np.median(tmp)])
    

result = np.stack(mean_cof)
plt.scatter(result[:,0], result[:,1])
    
    
    
    














