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
            
            print(raw_data[dtype]['data'].keys())
            
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


def one_hot_label_generator(labels, divisions = [0,5,10,20]):

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

def moving_average_label(data, period, steps):
    
    avg_data = []
    
    for i in range(period,len(data), steps):
      
        cat_label = np.concatenate(data[i-period:i])
        if len(cat_label) < 95:
            avg_data.append(-10)
        else:
            avg_data.append(np.mean(cat_label))
            
            
    
    return np.array(avg_data)



def generate_data_label_from_file(filename, preprocess_config, label_config):
    
    
    #train_label = []
    
    h5data = pd.HDFStore(filename)
    raw_data= h5data["raw_data"]
    train_data = preprocess_data(raw_data ,preprocess_config)
    
    #for device in label_config['raw_data_type']:

        #train_label.append(raw_data[device]['label'][label_config['label_feature']])
    
    train_label = np.array(raw_data['AP']['label'][label_config['label_feature']])
    
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

"""
data_dict = {   "data":{
                    "Rcv":[],
                    "Rx_bitrate":[],
                    "Tx_bitrate":[],
                    "SS_Sigval":[],
                    "SS_Sigval_Std":[],
                    "SS_Portion":[],
                    "SS_Count":[],
                    "SS_Rssi":[],
                    "FCSError":[],
                    "CRC-ERR":[],
                    "LENGTH-ERR":[],
                    "PHY-ERR":[]
                },

                "label":{
                    "Ping_mean":[],
                    #"Ping_std":[],
                    "FER":[]
                }
                
            }
"""

preprocess_config = {
        
            'feature_type' : ['SS_Sigval_Std', 'SS_Sigval', 'SS_Portion', 'FCSError', 'Rcv'], 
            'raw_data_type' : ['AP'], 
            'is_norm': True,
            'is_fft' : False
        
        }

label_config = {
        
            'label_feature' : 'Ping_mean', 
            'raw_data_type' : ['AP'], 
            
        
        }

def concat_data_label(data_path, file_list,preprocess_config, label_config):
    
    data = []
    label = []
    
    for f in file_list:
        
        fpath = os.path.join(data_path, f+'.h5')
        tmp_data, tmp_label = generate_data_label_from_file(fpath, preprocess_config, label_config) 
        
        if len(data) == 0:
            
            data = tmp_data
            label = tmp_label
        else:
            data = np.concatenate([data,tmp_data], axis=0)
            label = np.concatenate([label,tmp_label], axis=0)
            
    return data, label
        
"""
collections = ['1070202small-t1', '1070202small-t1-2', 
               '1070201small-t1', '1070201small-t1-2',
               '1070201small-t3', '1070201small-t3-4', '1070201small-t3-5'
               ]
"""
train_data_path = '../data/ProcessData1070208/'
train_data = ['1070208small-t2', '1070208small-t3']
test_data_path = '../data/ProcessData1070208/'
test_data = ['1070208small-t2-2', '1070208small-t3-2']


moving_average_window = 10
moving_average_step = 1

train_data, train_label =  concat_data_label(train_data_path, train_data, preprocess_config, label_config) 
test_data, test_label = concat_data_label(test_data_path, test_data, preprocess_config, label_config)    

train_data = moving_average(train_data, moving_average_window, moving_average_step)
train_label = moving_average_label(train_label, moving_average_window, moving_average_step)
train_label_oh = one_hot_label_generator(train_label)

test_data = moving_average(test_data, moving_average_window, moving_average_step)
test_label = moving_average_label(test_label, moving_average_window, moving_average_step)
test_label_oh = one_hot_label_generator(test_label)


import xgboost as xgb
from xgboost import plot_tree
from xgboost import plot_importance

#train_label_mask = np.where(np.array(train_label_oh) == 0, 0 ,1)

model = xgb.XGBClassifier(max_depth=5, learning_rate=0.01 ,n_estimators=2000, silent=False)
model.fit(train_data, train_label_oh)

xg_prediction_train = model.predict(train_data)

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



















        