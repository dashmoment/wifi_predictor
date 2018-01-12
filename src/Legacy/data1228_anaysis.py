
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import os
import random
#import model_zoo as mz

#import tensorflow.contrib.eager as tfe
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import copy


def preprocess_data(raw_data, feature_type = ['sigval', 'busy_time', 'Rssi'], raw_data_type = ['STA', 'AP']):
    
    data = []
    
    for i in range(len(feature_type)):
        
        tmp_data = raw_data[raw_data_type[0]][feature_type[i]] + raw_data[raw_data_type[1]][feature_type[i]] 
        tmp_data = np.around(np.array(tmp_data).astype(np.float32), decimals=3)
         
        flat_data =tmp_data.reshape(-1,1)
        norm = np.mean(flat_data, axis=0)
        std = np.std(flat_data, axis=0)
        tmp_data = (tmp_data - norm)/std
        
        if len(tmp_data.shape) != 3: tmp_data = tmp_data.reshape(-1,20,1)
        
        data.append(tmp_data)
        
    return np.concatenate(data, axis=-1)

data_feature = 'sigval'
label_feature = 'FER'
#h5data = pd.HDFStore('../data/ProcessData1228/testing_data.h5')
h5data = pd.HDFStore('../data/ProcessData1228/training_data.h5')
raw_data= h5data["raw_data"]

slice_data = preprocess_data(raw_data)
slice_label = raw_data['STA'][label_feature] + raw_data['AP'][label_feature]
raw_data = None

new_label = []
for i in range(len(slice_label)): 
    
    tmp = int(slice_label[i]//0.2)
    if tmp < 2: new_label.append(int(slice_label[i]//0.2)) 
    else: new_label.append(2)
    
slice_label  = copy.deepcopy(np.array(new_label)) 
hist = np.histogram(slice_label, bins=[0,0.9, 1.2, 2.2, 3.2,4.2])

"""
h5data = pd.HDFStore('../data/ProcessData1228/testing_data.h5')
raw_data= h5data["raw_data"]
testslice_data = preprocess_data(raw_data)
testslice_label = raw_data['STA'][label_feature] + raw_data['AP'][label_feature]
testslice_label = testslice_label
raw_data = None    
"""
