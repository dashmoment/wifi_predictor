import mongodb_api as db
import pandas as pd
import sys
import numpy as np
from sklearn.svm import SVR

from matplotlib import pyplot as plt

m = db.mongodb_api(user='ubuntu', pwd='ubuntu', collection="TestData1213")
found_data = m.find(ftype='many')
_found_data = {} 

idx = 0

survey_dict = {
               "survey_data":[],
               "survey_mean":[],
               "survey_std":[],
               "survey_FER":[]
              }

spectralscan_data_dict = {
                         "Spectralscan_data":[],
                         "Spectralscan_mean":[],
                         "Spectralscan_std":[],
                         "Spectralscan_FER":[]
                          }

survey_label = []
Spectralscan_label = []
survey_station_info = []
Spectralscan_station_info = []




def label_generator(labels):
    
    labels_mean = np.mean(np.array(labels))
    labels_std = np.std(np.array(labels))

    return labels_mean, labels_std


found_data_dict = {}

for i in range(len(found_data)):
    
    key = list(found_data[i].keys())[0]
    found_data_dict[key] = found_data[i][key]
found_data_1 = found_data_dict

for key in sorted(found_data_dict):
    
    if key=="_id":
        continue
    
    value = found_data[key]
    
    if(value["Topic"]=="Survey"):
        
        if len(survey_label) > 0:
            
            survey_data = [value["Busy(ms)"],value["Rcv(ms)"], value["Tx(ms)"]]
            #survey_data = [value["Busy(ms)"],value["Rcv(ms)"]]
            label_val = label_generator(survey_label)
            
            survey_dict["survey_data"].append(survey_data)
            survey_dict["survey_mean"].append(label_val[0]) 
            survey_dict["survey_std"].append(label_val[1]) 
            survey_dict["survey_FER"].append(survey_station_info) 
            
            survey_label = []
            survey_station_info = []
    
    elif(value["Topic"]=="Spectralscan"):
        
        
        
        if len(Spectralscan_label) > 1:
            print("{} Processed: {}/{}".format(len(survey_label),idx, len(found_data)))
            sigval = value["Sigval"]
            label_val = label_generator(Spectralscan_label)
            spectralscan_data_dict["Spectralscan_data"].append(sigval)
            spectralscan_data_dict["Spectralscan_mean"].append(label_val[0]) 
            spectralscan_data_dict["Spectralscan_std"].append(label_val[1]) 
            spectralscan_data_dict["Spectralscan_FER"].append(Spectralscan_station_info)
            
            Spectralscan_station_info = []     
            Spectralscan_label = []
        
    
    elif value["Topic"]=="Ping":
        
       tmp_label = value["Delay(ms)"]
       survey_label.append(tmp_label)
       Spectralscan_label.append(tmp_label)
       
    elif value["Topic"]=="Stationinfo":
        
       tmp_label = value["FER"]
       Spectralscan_station_info.append(tmp_label)
       survey_station_info.append(tmp_label)
    
    idx += 1
    

  
survey_data  = pd.DataFrame.from_dict(survey_dict)
scan_data  = pd.DataFrame.from_dict(spectralscan_data_dict)


#store = pd.HDFStore('../data/survey_data.h5')
#store['raw_data'] = survey_data
#
#store = pd.HDFStore('../data/scan_data.h5')
#store['raw_data'] = scan_data













