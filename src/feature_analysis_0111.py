import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import os
import random
#import model_zoo as mz

#import tensorflow.contrib.eager as tfe
from sklearn.metrics import confusion_matrix
from scipy.fftpack import fft
from matplotlib import pyplot as plt
import copy
#import mongodb_api as db
import model_zoo as mz

# Training data
#['sigval', 'busy_time', 'Rssi', 'Portion']
#['sigval_std','sigval', 'busy_time', 'Rssi', 'Portion']


def norm_unistd(data):
    
    norm = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    n_data = (data - norm)/std
    
    return n_data
    
    
def preprocess_data(raw_data, kwarg):
    
    feature_type = kwarg['feature_type']
    raw_data_type = kwarg['raw_data_type']
    is_norm = kwarg['is_norm']
    is_fft = kwarg['is_fft']
    
    data = []
    
    
    for dtype in raw_data_type:
        
        devices_data = []
    
        for i in range(len(feature_type)):
            
            tmp_data = raw_data[dtype]['data'][feature_type[i]]
            tmp_data = np.around(np.array(tmp_data).astype(np.float32), decimals=3)
            tmp_data = tmp_data.reshape(-1,np.shape(tmp_data)[-1])
            
            if  is_norm and feature_type[i] != 'Portion':
                
                tmp_data = norm_unistd(tmp_data)        
            
            if is_fft and (feature_type[i] == 'Sigval' or feature_type[i] == 'Sigval_Std'):
        
                tmp_data = fft(tmp_data)
                
                Nfeature = 20
                tmp_data = tmp_data[:,:Nfeature].reshape(-1,Nfeature)           
            
            
            devices_data.append(tmp_data)
        
        data.append(np.concatenate(devices_data, -1))
    return np.concatenate(data,0)


def one_hot_label_generator(labels, divisions = [0,6,11,21]):

    def check_in_range(label, divisions):

        if label>=divisions[0] and label < divisions[1]:
            return True
        else:
            return False

    
    oh_label = []
    for l_idx in range(len(labels)):
        for idx in range(len(divisions)-1):
            
            if check_in_range(labels[l_idx],[divisions[idx], divisions[idx+1]]): 
                oh_label.append(idx)
                break
        if len(oh_label) != l_idx + 1:
            oh_label.append(len(divisions)-1)

       
    return oh_label


###Moving average

def moving_average(data, period, steps):
    
    avg_data = []
    for i in range(period,len(data), steps):
            avg_data.append(np.mean(data[i-period:i], axis=0))
            
            
    
    return np.array(avg_data)


def generate_data_label_from_file(filename, preprocess_config, label_config):
    
    
    train_label = []
    
    h5data = pd.HDFStore(filename)
    raw_data= h5data["raw_data"]
    train_data = preprocess_data(raw_data ,preprocess_config)
    
    for device in label_config['raw_data_type']:
        
        

        train_label.append(raw_data[device]['label'][label_config['label_feature']])
    
    train_label = np.array(train_label).reshape(-1,1)
    
    return train_data, train_label


"""
data_dict = {   "data":{
                    "Busy(ms)":[],
                    "Rx_Rate":[],
                    "Tx_Rate":[],
                    "Sigval":[],
                    "Sigval_Std":[],
                    "Portion":[],
                    "Count":[],
                    "Rssi":[],
                    "FCSError":[]
                },

                "label":{
                    "Ping_mean":[],
                    "Ping_std":[],
                    "FER":[]
                }
                
            }
"""


preprocess_config = {
        
            'feature_type' : ["Busy(ms)", "Sigval", "Portion", "FCSError", "Rssi"], 
            'raw_data_type' : ['STA'], 
            'is_norm': True,
            'is_fft' : False
        
        }

label_config = {
        
            'label_feature' : 'Ping_mean', 
            'raw_data_type' : ['STA'], 
            
        
        }

#train_data, train_label = generate_data_label_from_file('../data/ProcessData1070110/training_data_mid_t1.h5', preprocess_config, label_config)
#test_data, test_label = generate_data_label_from_file('../data/ProcessData1070110/testing_data_mid_t1.h5', preprocess_config, label_config)

#train_data, train_label = generate_data_label_from_file('../data/ProcessData1070110/training_data_nsmall_t1.h5', preprocess_config, label_config)
#test_data, test_label = generate_data_label_from_file('../data/ProcessData1070110/testing_data_nsmall_t1.h5', preprocess_config, label_config)

#train_data, train_label = generate_data_label_from_file('../data/ProcessData1070110/training_data_t1.h5', preprocess_config, label_config)
#test_data, test_label = generate_data_label_from_file('../data/ProcessData1070110/testing_data_t1.h5', preprocess_config, label_config)

path = '../data/ProcessData1070112/training_data_mid_'
train_data = []
train_label = []
Ntraining_set = 4
for n in range(Ntraining_set):
    
    filename = path + str(n+1) + '.h5'
    tmp_data, tmp_label = generate_data_label_from_file(filename, preprocess_config, label_config)   
    
    if n == 0:
        train_data = tmp_data
        train_label = tmp_label
    
    else:
        train_data = np.concatenate([train_data,tmp_data], axis=0)
        train_label = np.concatenate([train_label,tmp_label], axis=0)

test_data, test_label = generate_data_label_from_file('../data/ProcessData1070112/training_data_mid_5.h5', preprocess_config, label_config)    

train_data = moving_average(train_data, 10, 1)
train_label = moving_average(train_label, 10, 1)
train_label_oh = one_hot_label_generator(train_label)

test_data = moving_average(test_data, 10, 1)
test_label = moving_average(test_label, 10, 1)
test_label_oh = one_hot_label_generator(test_label)




import xgboost as xgb
from xgboost import plot_tree
from xgboost import plot_importance

model = xgb.XGBClassifier(max_depth=5, learning_rate=0.01 ,n_estimators=2000, silent=False)
model.fit(train_data, train_label_oh)

#plot_tree(model)
#plt.show()
#
xg_prediction_train = model.predict(train_data)
xg_accuracy_train = np.mean(np.equal(train_label_oh, xg_prediction_train).astype(np.float32))
xg_train_c_matrix = confusion_matrix(train_label_oh, xg_prediction_train)
#plt.imshow(xg_train_c_matrix)
#plt.show()

xg_prediction_test = model.predict(test_data)
xg_accuracy_test = np.mean(np.equal(test_label_oh, xg_prediction_test).astype(np.float32))
xg_test_c_matrix = confusion_matrix(test_label_oh, xg_prediction_test)
#plt.imshow(xg_test_c_matrix)
#plt.show()


plot_importance(model)
plt.show()



















        