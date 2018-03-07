import mongodb_api as db
import pandas as pd
import sys
import numpy as np
from sklearn.svm import SVR
import random
import os
from matplotlib import pyplot as plt
import copy

def label_generator(labels, axis=0):
    
    labels_mean = np.mean(np.array(labels), axis = axis)
    labels_std = np.std(np.array(labels), axis = axis)

    return labels_mean, labels_std

### Process Data Format like
#m = db.mongodb_api(user='ubuntu', pwd='ubuntu', collection="ProcessData1213")
#found_data = m.find(ftype='many')



data_dict = {   #Data
                "busy_time":[],
                "rx_rate":[],
                "tx_rate":[],
                "sigval":[],
                "sigval_std":[],
                "Portion":[],
                "Count":[],
                "Rssi":[],
                "FCSError":[],
                #label
                "Ping_mean":[],
                "Ping_std":[],
                "FER":[],
                
                          }


train_data = {'AP':copy.deepcopy(data_dict), 'STA':copy.deepcopy(data_dict)}
test_data = {'AP':copy.deepcopy(data_dict), 'STA':copy.deepcopy(data_dict)}
devices_type = ['AP','STA']



#m = db.mongodb_api(user='ubuntu', pwd='ubuntu', collection="ProcessData1061228")
m = db.mongodb_api(user='ubuntu', pwd='ubuntu', collection="1070208small-t1")
found_data = m.find(ftype='many')
output_folder = '../data/ProcessData1070208'
output_file = 'training_data_mid_5.h5'
output_path = os.path.join(output_folder,output_file)

if not os.path.isdir(output_folder):
    os.mkdir(output_folder)


training_portion = 0.0 #Portion for training data

time_step = 1
time_stride = 1
train_pairs = []
test_pairs = []

for idxs in range(0,(len(found_data) - time_step), time_stride):
#for idxs in range(0,500, time_stride):
    
    print(idxs)
    
    rndnum = random.randrange(0, 10)
    container = train_data
    """
    if len(test_data[devices_type[0]]['Count']) < 0.2*len(found_data):
        if rndnum >= 5 : container = train_data
        else: container = train_data
    else:
        container = train_data
    """
    for device in devices_type:
    
        tmp_busy_time = []
        tmp_Rx_Rate = []
        tmp_Tx_Rate = []
        tmp_sigval = []
        tmp_sigval_std = []
        tmp_Portion = []
        tmp_RSSI = []
        
        tmp_count = []
        tmp_Ping_mean = []
        tmp_Ping_std = []
        tmp_FER = []
        tmp_Count = []
        
        save = 1
        
        for i in range(time_step):
            
            cur_idx = idxs + i
            
            if len(found_data[cur_idx][device]["Sigval"]) == 0 : 
                save = 0
                break
            
            
            tmp_busy_time.append(found_data[cur_idx][device]['Busy(ms)'])
            tmp_Rx_Rate.append(found_data[cur_idx][device]['Rx_Rate'])
            tmp_Tx_Rate.append(found_data[cur_idx][device]['Tx_Rate'])
            
            #assert len(found_data[cur_idx][device]["Sigval"]) == 1 or len(found_data[cur_idx][device]["Sigval"]) == 2
            
            if len(found_data[cur_idx][device]["Sigval"]) == 1: 
                
                genlabel = lambda x: np.array(x) 
            else:
                genlabel = label_generator
                
            tmp_sigval.append(genlabel(found_data[cur_idx][device]['Sigval'])[0]) 
            tmp_sigval_std.append(genlabel(found_data[cur_idx][device]['Sigval_Std'])[0]) 
            tmp_RSSI.append(genlabel(found_data[cur_idx][device]['Rssi'])[0])
            tmp_Portion.append(genlabel(found_data[cur_idx][device]['Portion'])[0]) 
            
#            tmp_count.append(len(found_data[cur_idx][device]['Count']))
#            tmp_FER.append(found_data[cur_idx][device]['FER'])
#            tmp_Ping_mean.append(genlabel(found_data[cur_idx][device]['Ping'])[0])
#            tmp_Ping_std.append(genlabel(found_data[cur_idx][device]['Ping'])[1])
        
        if save == 1:
        
            container[device]["busy_time"].append(tmp_busy_time)
            container[device]["rx_rate"].append(tmp_Rx_Rate)
            container[device]["tx_rate"].append(tmp_Tx_Rate)
            container[device]["sigval"].append(tmp_sigval)
            container[device]["sigval_std"].append(tmp_sigval_std)
            container[device]["Portion"].append(tmp_Portion)
            container[device]["Rssi"].append(tmp_RSSI)
            
            if len(found_data[cur_idx][device]['Ping']) > 0: 
                container[device]["Ping_mean"].append(label_generator(found_data[cur_idx][device]['Ping'])[0])
                container[device]["Ping_std"].append(label_generator(found_data[cur_idx][device]['Ping'])[1])
            
            else:
                container[device]["Ping_mean"].append(1000)
                container[device]["Ping_std"].append(1000)
            container[device]["FER"].append(found_data[cur_idx][device]['FER'])
            container[device]["Count"].append(len(found_data[cur_idx][device]['Count']))

 
   
        
processed_data  = pd.DataFrame.from_dict(train_data)
store = pd.HDFStore(output_path)
store['raw_data'] = processed_data

     
        
        
        
        
        
        
        
        
        
        
        
        
        
        