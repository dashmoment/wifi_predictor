# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 11:34:39 2018

@author: MaggieYC_Pang
"""

import sys
sys.path.append("../")

from mongodb_api import mongodb_api
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def moving_func(ip_list, step, func=np.mean, arg=None):
    op_list=[]
    i=0
    for data in ip_list[step:]:
        op_list.append(func(ip_list[i:i+step], arg))
        i=i+1
    return op_list


class wifi_diag_api:    
    
    def __init__(self):
        
        label_list = []
        label_index_dict = {}
        topicdata_dict = {}
        
        # ============================== ML OUTPUT ===============================
        
        label_list.append({"Name":"Delay", "Topic":"Ping", "MLType":"Out", "Process":[np.mean, np.std, len]})   
        label_list.append({"Name":"Tput", "Topic":"Iperf", "MLType":"Out", "Process":[np.mean]})    
        label_list.append({"Name":"Jitter", "Topic":"Iperf", "MLType":"Out", "Process":[np.mean]})    
        label_list.append({"Name":"Loss", "Topic":"Iperf", "MLType":"Out", "Process":[np.mean]})    
        
        label_list.append({"Name":"Tx_bitrate", "Topic":"Stationinfo", "MLType":"Out"})    
        label_list.append({"Name":"Rx_bitrate", "Topic":"Stationinfo", "MLType":"Out"})    
        label_list.append({"Name":"Signal", "Topic":"Stationinfo", "MLType":"Out"})    
        label_list.append({"Name":"FER", "Topic":"Stationinfo", "MLType":"Out"})    
        
        # ============================== ML INPUT ===============================
        
        label_list.append({"Name":"SS_Sigval",      "Topic":"Spectralscan", "MLType":"In", "Process":[np.array]})    
        label_list.append({"Name":"SS_Sigval_Std",  "Topic":"Spectralscan", "MLType":"In", "Process":[np.array]})    
        label_list.append({"Name":"SS_Portion",     "Topic":"Spectralscan", "MLType":"In", "Process":[np.array]})    
        label_list.append({"Name":"SS_Count",       "Topic":"Spectralscan", "MLType":"In", "Process":[np.sum]})
        label_list.append({"Name":"SS_Rssi",        "Topic":"Spectralscan", "MLType":"In", "Process":[np.mean]})    
        label_list.append({"Name":"SS_Noise",       "Topic":"Spectralscan", "MLType":"In", "Process":[np.mean]})    
        
        label_list.append({"Name":"Busy",   "Topic":"Survey", "MLType":"In"})    
        label_list.append({"Name":"Noise",  "Topic":"Survey", "MLType":"In"})    
        label_list.append({"Name":"Rcv",    "Topic":"Survey", "MLType":"In"})    
        label_list.append({"Name":"Tx",     "Topic":"Survey", "MLType":"In"})    
        
        label_list.append({"Name":"FCSError", "Topic":"Statistics", "MLType":"In"})
        
        ERR_list = ["CRC-ERR", "LENGTH-ERR", "PHY-ERR", "SPECTRAL"] # USEFUL
    #    ERR_list = ["CRC-ERR", "DECRYPT-BUSY-ERR", "DECRYPT-CRC-ERR", "LENGTH-ERR", "MIC-ERR", "OOM-ERR", "PHY-ERR", "POST-DELIM-CRC-ERR", "PRE-DELIM-CRC-ERR", "RATE-ERR", "SPECTRAL"]
        for data in ERR_list:
            label_list.append({"Name":data, "Topic":"ath9kERR", "MLType":"In"})
            
            
    #    ERR_list = ["chan_idle_dur", "chan_idle_dur_valid", "dcu_arb_state", "dcu_complete_state", "dcu_fp_state", 
    #                "qcu_complete_state", "qcu_fetch_state", "qcu_stitch_state", 
    #                "txfifo_dcu_num_0", "txfifo_dcu_num_1", "txfifo_valid_0", "txfifo_valid_1"]
        ERR_list = ["chan_idle_dur", "chan_idle_dur_valid"] #USEFUL
        for data in ERR_list:
            label_list.append({"Name":data, "Topic":"ath9kDMA", "MLType":"In"})
                    
    #    ERR_list = ["ANI_RESET", "CCK_ERRORS", "CCK_LEVEL", "FIR-STEP_DOWN", "FIR-STEP_UP", "INV_LISTENTIME", "MRC-CCK_OFF", "MRC-CCK_ON",
    #                                 "OFDM_ERRORS", "OFDM_LEVEL", "OFDM_WS-DET_OFF", "OFDM_WS-DET_ON", "SPUR_DOWN", "SPUR_UP"]
        ERR_list = ["CCK_ERRORS", "OFDM_ERRORS", "SPUR_DOWN", "SPUR_UP"] #USEFUL
        for data in ERR_list:
            label_list.append({"Name":data, "Topic":"ath9kANI", "MLType":"In"})
        
        # ============================== END ===============================
               
        for labeldata in label_list:
            label_index_dict[labeldata["Name"]] = label_list.index(labeldata)
            label_index_dict[label_list.index(labeldata)] = labeldata["Name"]
            
            if(labeldata["Topic"] not in topicdata_dict):
                topicdata_dict[labeldata["Topic"]]=[]
            
            if("Process" not in labeldata):
                topicdata_dict[labeldata["Topic"]].append([labeldata["Name"], "single"])
            else:
                topicdata_dict[labeldata["Topic"]].append([labeldata["Name"], "list"])      
        
        # =========================================================================================================  
        
        process_name_dict={}
        process_name_dict[np.mean] = "mean"
        process_name_dict[np.std] = "std"
        process_name_dict[np.sum] = "sum"
        process_name_dict[np.array] = "array"
        process_name_dict[len] = "len"
        
        self.label_list = label_list
        self.process_name_dict = process_name_dict
        self.label_index_dict = label_index_dict
        self.topicdata_dict = topicdata_dict

    def GetDataList(self, dev, found_data, name, proc):
        retlist = []
        for data in found_data:
            target = data[dev]
                
            if(name not in target):
                retlist.append(-1)
            else:
                if(proc==None):
                    retlist.append(target[name])
                else:
                    retlist.append(proc(target[name]))
        return retlist
    
    def plot_all(self, mdb):
        
        print("collection = " + mdb.get_full_name())
        found_data = mdb.find(key_value = {}, ftype='many')
        print("len(found_data) = " + str(len(found_data)))
        
        ML_data_AP = {}   
        ML_data_STA = {}   
                      
        for labeldata in self.label_list:        
            if(labeldata["Name"] not in found_data[0]["AP"]):
                continue
            
            if("Process" not in labeldata):
                ML_data_AP[labeldata["Name"]] = self.GetDataList("AP", found_data, labeldata["Name"], None)
            else:
                for proc in labeldata["Process"]:
                    ML_data_AP[labeldata["Name"] + '_' + self.process_name_dict[proc]] = self.GetDataList("AP", found_data, labeldata["Name"], proc)
            
            if("Process" not in labeldata):
                ML_data_STA[labeldata["Name"]] = self.GetDataList("STA", found_data, labeldata["Name"], None)
            else:
                for proc in labeldata["Process"]:        
                    ML_data_STA[labeldata["Name"] + '_' + self.process_name_dict[proc]] = self.GetDataList("STA", found_data, labeldata["Name"], proc)
                            
        for pkey in ML_data_AP:
            
            if("array" in pkey):
                continue
            
            plt.plot(moving_func(ML_data_AP[pkey],10), 'b.')
            plt.plot(moving_func(ML_data_STA[pkey],10), 'g.')
            plt.show()
            print("pkey: " + pkey)
              
        APdf = pd.DataFrame(ML_data_AP)
        STAdf = pd.DataFrame(ML_data_STA)
        
        return APdf, STAdf
        
    def create_df(self, mdb, step=1, func=np.mean, arg=None):
        
        print("collection = " + mdb.get_full_name())
        found_data = mdb.find(key_value = {}, ftype='many')
        print("len(found_data) = " + str(len(found_data)))
        
        ML_data = {}
                      
        for labeldata in self.label_list:        
            if(labeldata["Name"] not in found_data[0]["AP"]):
                continue
            
            if("Process" not in labeldata):
                ML_data["AP-" + labeldata["Name"]] = self.GetDataList("AP", found_data, labeldata["Name"], None)
            else:
                for proc in labeldata["Process"]:
                    ML_data["AP-" + labeldata["Name"] + '_' + self.process_name_dict[proc]] = self.GetDataList("AP", found_data, labeldata["Name"], proc)
                    
            if("Process" not in labeldata):
                ML_data["STA-" + labeldata["Name"]] = self.GetDataList("STA", found_data, labeldata["Name"], None)
            else:
                for proc in labeldata["Process"]:        
                    ML_data[ "STA-" + labeldata["Name"] + '_' + self.process_name_dict[proc]] = self.GetDataList("STA", found_data, labeldata["Name"], proc)
                                          
        df = pd.DataFrame(ML_data)
        return df    
    
    def classification(self, ML_data):
        
        classify_dict={}
        classify_dict["AP-Delay_mean"] = [5,10,20]    
        classify_dict["AP-Delay_len"] = [7,8,9,9.5,10]
        
        classify_result={}
        
        for ckey, cdata in classify_dict.items():
            
            target_list = ML_data[ckey]
            classify_result[ckey]=[]
            for target in target_list:
                index = 0
                while (index < len(cdata)):
                    if(target < cdata[index]):
                        break
                    index = index + 1
                classify_result[ckey].append(index)
        
        fclass = [0,0,0,0,0]
        for index in range(len(classify_result["AP-Delay_mean"])):
            if(classify_result["AP-Delay_len"][index] < 4):
                fclass[4] = fclass[4]+1
            else:
                fclass[classify_result["AP-Delay_mean"][index]] = fclass[classify_result["AP-Delay_mean"][index]]+1
            
        class_df = pd.DataFrame(classify_result)
        
        return class_df, fclass