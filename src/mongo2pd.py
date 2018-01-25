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


train_data = {'AP':copy.deepcopy(data_dict), 'STA':copy.deepcopy(data_dict)}
devices_type = ['AP','STA']



m = db.mongodb_api(user='ubuntu', pwd='ubuntu', collection="ProcessDataPair1070112mid-5")
found_data = m.find(ftype='many')
output_folder = '../data/ProcessData1070112'
output_file = 'training_data_mid_5.h5'
output_path = os.path.join(output_folder,output_file)

if not os.path.isdir(output_folder):
    os.mkdir(output_folder)

time_step = 1
time_stride = 1

def check_valid_data(cur_idx, time_step, raw_data, device):

    for i in range(cur_idx, cur_idx+time_step):
        
        if len(raw_data[i][device]["Sigval"]) == 0:
            return False

    if len(found_data[cur_idx+time_step][device]["Ping"]) <= 0:
        return False

    return True

for idxs in range(0,(len(found_data) - time_step), time_stride):
#for idxs in range(0,500, time_stride):
    
    print("Processed:{}/{}".format(idxs, (len(found_data) - time_step)//time_stride))
    
    container = train_data
    
    for device in devices_type:
        
        if check_valid_data(idxs, time_step, found_data, device):

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
        
        
        
        container[device]['label']["Ping_mean"].append(label_generator(found_data[cur_idx][device]['Ping'])[0])
        container[device]['label']["Ping_std"].append(label_generator(found_data[cur_idx][device]['Ping'])[1])
        container[device]['label']["FER"].append(found_data[cur_idx][device]['FER'])
        
   
processed_data  = pd.DataFrame.from_dict(train_data)
store = pd.HDFStore(output_path)
store['raw_data'] = processed_data

     
        
        
        
        
        
        
        
        
        
        
        
        
        
        