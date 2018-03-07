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







def check_valid_data(found_data, cur_idx, time_step, raw_data, device):

    for i in range(cur_idx, cur_idx+time_step):
        
        if len(raw_data[i][device]["SS_Sigval"]) == 0:
            return False

    if len(found_data[cur_idx+time_step][device]["Delay"]) <= 0:
        return False

    return True

def gnerate_data_and_label(found_data, data_dict, time_step=1, time_stride=1):
    
    train_data = {'AP':copy.deepcopy(data_dict), 'STA':copy.deepcopy(data_dict)}
    devices_type = ['AP','STA']

    for idxs in range(0,(len(found_data) - time_step), time_stride):
    
    
    #for idxs in range(0,500, time_stride):
        
        print("Processed:{}/{}".format(idxs, (len(found_data) - time_step)//time_stride))
        
        container = train_data
        
        for device in devices_type:
            
            
            if check_valid_data(found_data, idxs, time_step, found_data, device):
    
                tmp_data = copy.deepcopy(data_dict)
    
                for i in range(time_step):
    
                    cur_idx = idxs + i
    
                    ###Get each time step data, special handle count
    
                    for key in tmp_data['data']:
                        
                        if key == 'Count':
                           
                            tmp_data['data'][key].append(len(found_data[cur_idx][device][key]))
                            
                        else:
    
                            #### Special hadle case of one "Sigval" values in each time step                     
                            if type(found_data[cur_idx][device][key]) != list or len(found_data[cur_idx][device][key]) == 1 : 
                                
                                genlabel = lambda x: np.array(x) 
                            else:
                                genlabel = label_generator
    
                        
                            gen_data = genlabel(found_data[cur_idx][device][key])
                            
                            if len(np.shape(gen_data)) == 0:
                                tmp_data['data'][key].append(genlabel(found_data[cur_idx][device][key]))
                            else:
                                tmp_data['data'][key].append(genlabel(found_data[cur_idx][device][key])[0])
    
            else:
                continue
            
            for k in container[device]['data']:
               
                container[device]['data'][k].append(np.array(tmp_data['data'][k]).reshape(time_step, -1))
            
            container[device]['label']["Ping_mean"].append(np.array(found_data[cur_idx][device]['Delay']))
            #container[device]['label']["Ping_std"].append(label_generator(found_data[cur_idx][device]['Delay'])[1])
            container[device]['label']["FER"].append(np.array(found_data[cur_idx][device]['FER']))
                
    return train_data
        


output_folder = '../data/ProcessData1070208'
collections = ['1070222-clear-ProcessData']
"""
collections = ['1070208small-t1', '1070208small-t1-2', 
               '1070208small-t2', '1070208small-t2-2',
               '1070208small-t3', '1070208small-t3-2'
               ]
"""
for c in collections:
    
    m = db.mongodb_api(user='ubuntu', pwd='ubuntu', database='wifi_diagnosis', collection=c+'-MLData')
    found_data = m.find(ftype='many')
    output_file = c +'.h5'
    output_path = os.path.join(output_folder,output_file)
    
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    
    time_step = 1
    time_stride = 1
    
    train_data = gnerate_data_and_label(found_data, data_dict) 
    processed_data  = pd.DataFrame.from_dict(train_data)
    processed_data.to_hdf(output_path, 'raw_data', mode='w')
    
    
    
    """
    
    store = pd.HDFStore(output_path, 'w')
    store.append('raw_data', processed_data)
    time.sleep(1)
    store.close
    """
     
     
        
        
        
        
        
        
        
        
        
        
        
        
        