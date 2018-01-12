import mongodb_api as db
import pandas as pd
import sys
import numpy as np
from sklearn.svm import SVR

from matplotlib import pyplot as plt



m = db.mongodb_api(user='ubuntu', pwd='ubuntu', collection="TestData1061205")
found_data = m.find(ftype='many')


processed_data = {}
key_list = []

for i in range(len(found_data)):
    
    key_list = list(found_data[i].keys())
    
    for k in key_list:
        processed_data[k] = found_data[i][k]
    

idx = 0

survey_dict = {
               "survey_data":[],
               "survey_mean":[],
               "survey_std":[]
              }


spectralscan_data_dict = {
                         "Spectralscan_data":[],
                         "Spectralscan_mean":[],
                         "Spectralscan_std":[],
                         "Sigval_Std":[],
                         "Portion":[]
                          }

survey_label = []
Spectralscan_label = []




def label_generator(labels):
    
    labels_mean = np.mean(np.array(labels))
    labels_std = np.std(np.array(labels))

    return labels_mean, labels_std

    

for key in sorted(processed_data):
    
    if key=="_id":
        continue
    
    value = processed_data[key]
    
    if(value["Topic"]=="Survey"):
        
        if len(survey_label) > 0 and value["Agent"]!="10.144.24.24":
            
            survey_data = [value["Busy(ms)"],value["Rcv(ms)"], value["Tx(ms)"]]
            #survey_data = [value["Busy(ms)"],value["Rcv(ms)"]]
            label_val = label_generator(survey_label)
            
            survey_dict["survey_data"].append(survey_data)
            survey_dict["survey_mean"].append(label_val[0]) 
            survey_dict["survey_std"].append(label_val[1]) 
            survey_label = []
    
    elif(value["Topic"]=="Spectralscan"):
        
              
        if len(Spectralscan_label) > 1:
            
            print("{} Processed: {}/{}".format(len(survey_label),idx, len(processed_data)))
            sigval = value["Sigval"]
            
            label_val = label_generator(Spectralscan_label)
            spectralscan_data_dict["Spectralscan_data"].append(sigval)
            spectralscan_data_dict["Spectralscan_mean"].append(label_val[0]) 
            spectralscan_data_dict["Spectralscan_std"].append(label_val[1])       
            spectralscan_data_dict["Sigval_Std"].append(value["Sigval_Std"])
            spectralscan_data_dict["Portion"].append(value["Portion"])
            
            Spectralscan_label = []
        
    
    elif value["Topic"]=="Ping" and value["Agent"]!="10.144.24.23":
        
       tmp_label = value["Delay(ms)"]
       survey_label.append(tmp_label)
       Spectralscan_label.append(tmp_label)
    
    idx += 1
    

  
survey_data  = pd.DataFrame.from_dict(survey_dict)
scan_data  = pd.DataFrame.from_dict(spectralscan_data_dict)


store = pd.HDFStore('../data/survey_data_1061205.h5')
store['raw_data'] = survey_data

store = pd.HDFStore('../data/scan_data_1061205.h5')
store['raw_data'] = scan_data













