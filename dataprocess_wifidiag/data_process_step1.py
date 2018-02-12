# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 09:57:20 2017

@author: MaggieYC_Pang
"""

import sys
sys.path.append("../")

from mongodb_api import mongodb_api
import matplotlib.pyplot as plt

def data_process(input_list , casetype):    
    if input_list is None:
        return None
    if not (type(input_list) is list):
        input_list = [input_list]
                
    
    output_dict = {}
    
    dataproc_dict = {}
    dataproc_dict["CwmData/Ping"] = [data_process_Ping, ""]
    dataproc_dict["CwmData/DevSurvey"] = [data_process_DevSurvey, ""]
    dataproc_dict["CwmData/DumpStation"] = [data_process_DumpStation, ""]
    dataproc_dict["CwmData/SpectralScan"] = [data_process_SpectralScan, ""]
    dataproc_dict["CwmData/DevScanDump"] = [data_process_DevScanDump, ""]
    dataproc_dict["CwmData/Iperf"] = [data_process_Ping, ""]    
    dataproc_dict["CwmData/DumpStatistics"] = [data_process_DumpStatistics, "Statistics"]
    dataproc_dict["CwmData/DumpAth9kRecv"] = [data_process_DumpStatistics, "ath9kERR"]
    dataproc_dict["CwmData/DumpAth9kDma"] = [data_process_DumpStatistics, "ath9kDMA"]
    dataproc_dict["CwmData/DumpAth9kAni"] = [data_process_DumpStatistics, "ath9kANI"]
    
    for input_data in input_list:            
        if input_data is None:
            continue
        
        if(input_data["Topic"] in dataproc_dict):
            if(dataproc_dict[input_data["Topic"]][1]==""):                
                new_data = dataproc_dict[input_data["Topic"]][0](input_data)
            else:
                new_data = dataproc_dict[input_data["Topic"]][0](input_data, dataproc_dict[input_data["Topic"]][1])
        
        elif(input_data["Topic"]=="CwmData/Tester"):
            if(casetype=="Start"):
                casetype = "Train"
            elif(casetype=="Train"):
                casetype = "Change"         
            elif(casetype=="Change"):
                casetype = "Test"           
            elif(casetype=="Test"):
                casetype = "End"           
            print("casetype = " + casetype)
            continue;
            
        else:
            continue;
            
        if type(new_data) is list:
            for data in new_data:
                output_dict.update(data)
        else:
            output_dict.update(new_data)

    return output_dict, casetype

def data_process_Ping(input_data):
    payload = input_data["Payload"]      
    new_list = []
    for ping_data in payload["Data"]:
        new_data = {}
        new_data_payload = {}
        
        new_data_payload["Topic"] = "Ping"
        new_data_payload["Agent"] = input_data["AgentName"]
        new_data_payload["Dst"] = payload["Target"]
        new_data_payload["Delay"] = ping_data[2]        
        
        Time = ping_data[0]
        Time = Time.replace(".",":")
#        Time = Time.replace(":"," ").replace("/"," ").split()
#        Time = [float(i) for i in Time]
        
        new_data[Time] = new_data_payload
        new_list.append(new_data)
        #new_data_payload["Time"] = Time
        #new_list.append(new_data_payload)
        
    return new_list        
    
def data_process_Iperf(input_data):
    payload = input_data["Payload"]      
    new_list = []
    for iperf_data in payload["Data"]:
        new_data = {}
        new_data_payload = {}
        
        new_data_payload["Topic"] = "Iperf"
        new_data_payload["Agent"] = input_data["AgentName"]
        new_data_payload["Src"] = payload["Target"]
        
        new_data_payload["Tput"] = iperf_data[1]
        new_data_payload["Jitter"] = iperf_data[2]
        new_data_payload["Loss"] = iperf_data[3]
        
        Time = iperf_data[0]
        Time = Time.replace(".",":")
#        Time = Time.replace(":"," ").replace("/"," ").split()
#        Time = [float(i) for i in Time]
        
        new_data[Time] = new_data_payload
        new_list.append(new_data)
        #new_data_payload["Time"] = Time
        #new_list.append(new_data_payload)
        
    return new_list        

def data_process_SpectralScan(input_data):    
    new_list = []
    for item in input_data["Payload"]:
        scandata = item["ScanData"]
        if len(scandata)==0:
            continue;
        new_data = {}
        new_data_payload = {}
        
        new_data_payload["Topic"] = "Spectralscan"
        new_data_payload["Agent"] = input_data["AgentName"]
        new_data_payload["DevName"] = input_data["DevName"]            
        new_data_payload["SS_Freq"] = scandata["Freq"]
        new_data_payload["SS_Rssi"] = scandata["Rssi"]
        new_data_payload["SS_Noise"] = scandata["Noise"]
        new_data_payload["SS_Sigval"] = scandata["Sigval"]
        new_data_payload["SS_Sigval_Std"] = scandata["Sigval_Std"]
        new_data_payload["SS_Portion"] = scandata["Portion"]
        new_data_payload["SS_Count"] = scandata["Count"]
        
        Time = item["Time"];
        Time = Time.replace(".",":")
#        Time = Time.replace(":"," ").replace("/"," ").split()
#        Time = [float(i) for i in Time]
        
        
        new_data[Time] = new_data_payload
        new_list.append(new_data)
        #new_data_payload["Time"] = Time
        #new_list.append(new_data_payload)
        
    return new_list

def data_process_DevSurvey(input_data):
    payload = input_data["Payload"]
    payload = payload[0]
    # Currently should only scan for one channel    
        
    new_data = {} 
    new_data_payload = {}
    new_data_payload["Topic"] = "Survey"
    new_data_payload["Agent"] = input_data["AgentName"]
    new_data_payload["Device"] = input_data["DevName"]
    
    for key in payload:
        new_data_payload[key] = payload[key]
    '''
    new_data_payload["Freq"] = payload["Freq"]
    new_data_payload["Active(ms)"] = payload["ActiveTime"]
    new_data_payload["Busy(ms)"] = payload["BusyTime"]
    new_data_payload["Rcv(ms)"] = payload["RcvTime"]
    new_data_payload["Tx(ms)"] = payload["TxTime"]   
    '''
    Time = input_data["Time"];
    Time = Time.replace(".",":")
#    Time = Time.replace(":"," ").replace("/"," ").split()
#    Time = [float(i) for i in Time]
    
    
    new_data[Time] = new_data_payload
    return new_data    
    #new_data_payload["Time"] = Time
    #return new_data_payload

def data_process_DumpStatistics(input_data, topicstr):
    payload = input_data["Payload"]
        
    new_data = {} 
    new_data_payload = {}
    new_data_payload["Topic"] = topicstr
    new_data_payload["Agent"] = input_data["AgentName"]
    new_data_payload["Device"] = input_data["DevName"]
    for key in payload:
        new_data_payload[key] = payload[key]
    
    Time = input_data["Time"];
    Time = Time.replace(".",":")
#    Time = Time.replace(":"," ").replace("/"," ").split()
#    Time = [float(i) for i in Time]
   
    new_data[Time] = new_data_payload
    return new_data    
    #new_data_payload["Time"] = Time
    #return new_data_payload    

def data_process_DumpStation(input_data):
    payload = input_data["Payload"]
    
    if(type(payload[0])==list):
        payload = payload[0]
        # Currently only one station          
        
    new_data = {} 
    new_data_payload = {}
    new_data_payload["Topic"] = "Stationinfo"
    new_data_payload["Agent"] = input_data["AgentName"]
    new_data_payload["AgentDev"] = input_data["DevName"]
    new_data_payload["TargetDev"] = payload[1]
    new_data_payload["Signal"] = payload[10]
    new_data_payload["Tx_bitrate"] = payload[11]
    new_data_payload["Rx_bitrate"] = payload[12]
    new_data_payload["Expected_tput"] = payload[13]
    if(payload[5]+payload[6]==0):
        new_data_payload["FER"]=0
    else:
        new_data_payload["FER"] = payload[6]/(payload[5]+payload[6])
        
    new_data_payload["RX_bytes"] = payload[2]
    new_data_payload["Rx_packets"] = payload[3]
    new_data_payload["TX_bytes"] = payload[4]
    new_data_payload["Tx_packets"] = payload[5]
    new_data_payload["Tx_retries"] = payload[6]
    new_data_payload["Tx_failed"] = payload[7]
    
    
    Time = payload[0];
 #   Time = input_data["Time"]
    Time = Time.replace(".",":")
#    Time = Time.replace(":"," ").replace("/"," ").split()
#    Time = [float(i) for i in Time]        
    
    new_data[Time] = new_data_payload
    return new_data    
    #new_data_payload["Time"] = Time
    #return new_data_payload
    
def data_process_DevScanDump(input_data):
    payload = input_data["Payload"]
            
    new_data_payload = {}
    new_data_payload["Topic"] = "Fingerprint"
    new_data_payload["Agent"] = input_data["AgentName"]
    
    location = payload["Location"].replace(","," ").replace("."," ").split()
    location = [float(i) for i in location]
    new_data_payload["Location"] = location
    
    scanlist = []
    for tmp in payload["Scan"]:
        if tmp=={}:
            continue
        if(tmp["MAC"]=="04:f0:21:23:8a:a1" or tmp["MAC"]=="04:f0:21:23:8a:9d" or tmp["MAC"]=="04:f0:21:23:8a:9f"):
            continue
        scanlist.append(tmp)
        
    new_data_payload["Scan"] = scanlist
    
    return new_data_payload    
    #new_data_payload["Time"] = Time
    #return new_data_payload
    
if __name__ == '__main__': 
    
    date='1070207small-t1'
    
    R_coll = date + '-RawData'
    W_coll = date + '-TestData' 
    
    m = mongodb_api(user='ubuntu', pwd='ubuntu', database='wifi_diagnosis', collection=R_coll)    
    
    
#    find_reg = {"Topic":"CwmData/DumpAth9kAni"}
    find_reg = {}
    found_data = m.find(key_value = find_reg, ftype='many')
    print(len(found_data))    
    
    
    
    m = mongodb_api(user='ubuntu', pwd='ubuntu', database='wifi_diagnosis', collection=W_coll)
    m.remove(key_value = {}, justone=False)
    
    casetype = "Start"
    for data in found_data:
        out, casetype = data_process(data,casetype)
        for key, value in out.items():
            value["Casetype"] = casetype
            m.insert({key:value})
    
    found_data2 = m.find(key_value = {}, ftype='many')
    print(len(found_data2))
    
    
    
        