# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 16:34:30 2018

@author: MaggieYC_Pang
"""

import sys
sys.path.append("../")

from mongodb_api import mongodb_api
from wifi_diag_api import wifi_diag_api
import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd

def Testdata_sort(mdb):
    
    found_data = mdb.find(key_value = {}, ftype='many')
    print(len(found_data))
    
    for data in found_data:
        data.pop("_id", None)        
        
    sorted_data = sorted(found_data, key=lambda k: list(k.keys())[0]) 
    return sorted_data
    
def data_process(wapi, sorted_data, mdb, iptable, skipfirstN=0):
#def data_process(topicdata_dict, sorted_data, mdb, iptable, skipfirstN=0):
    
    mdb.remove(key_value={}, justone=False)
    
    TestArr=[]
    
    A = sorted_data[skipfirstN:]
    
    cnt = 0
    typecnt=[0,0,0,0,0]
    testtype={"Start":0, "Train":1, "Change":2, "Test":3, "End":4}
    newdict={}
    
    for data in A:
        for key, value in data.items():
                if(cnt == 8):
                    if("FER" not in newdict["STA"]): #Disconnecting
                        print("F count=%d" % A.index(data))
                        print("testtype = " + newdict["Casetype"])
                        print("typecnt = %d" % typecnt[testtype[newdict["Casetype"]]])      
                        
                        for eachitem in wapi.topicdata_dict["Stationinfo"]:
                            if(eachitem[1]=="list"):
                                newdict["STA"][eachitem[0]].append(-1)
                            if(eachitem[1]=="single"):
                                newdict["STA"][eachitem[0]] = -1                      
                                
                    if("FER" not in newdict["AP"]):
                        print("G count=%d " % A.index(data))
                        print("testtype = " + newdict["Casetype"])
                        print("typecnt = %d" % typecnt[testtype[newdict["Casetype"]]])                        
                        
                        for eachitem in wapi.topicdata_dict["Stationinfo"]:
                            if(eachitem[1]=="list"):
                                newdict["AP"][eachitem[0]].append(-1)
                            if(eachitem[1]=="single"):
                                newdict["AP"][eachitem[0]] = -1
                                
                    if("Busy" not in newdict["STA"]):
                        print("H count=%d" % A.index(data))
                        print("testtype = " + newdict["Casetype"])
                        print("typecnt = %d" % typecnt[testtype[newdict["Casetype"]]])
                    if("Busy" not in newdict["AP"]):
                        print("I count=%d" % A.index(data))
                        print("testtype = " + newdict["Casetype"])
                        print("typecnt = %d" % typecnt[testtype[newdict["Casetype"]]])
                            
                    newdict["Time"]=key
                    typecnt[testtype[newdict["Casetype"]]] = typecnt[testtype[newdict["Casetype"]]] +1
#                    print(key + " : " + value["Casetype"])
                    TestArr.append(newdict)
                    mdb.insert(newdict)
                    newdict={}
                    cnt = 0
                        
                if(newdict=={}):        
                    newdict["STA"]={}
                    newdict["AP"]={}
                    newdict["Casetype"]=value["Casetype"]
                    
                    for proc_key, proc_value in wapi.topicdata_dict.items():
                        for eachitem in proc_value:
                            if(eachitem[1]=="list"):
                                newdict["STA"][eachitem[0]]=[]
                                newdict["AP"][eachitem[0]]=[]
                
                tmpkey = iptable[value["Agent"]]
                target = newdict[tmpkey]    
                
                for eachitem in wapi.topicdata_dict[value["Topic"]]:
#                    print("Topic: " + value["Topic"])
#                    print("Target: " + str(eachitem))
#                    print("value: ")
#                    print(value)
                    if(eachitem[1]=="list"):
                        target[eachitem[0]].append(value[eachitem[0]])
                    if(eachitem[1]=="single"):
                        target[eachitem[0]] = value[eachitem[0]]                        
                    
                if(value["Topic"] == "Stationinfo"):
                    #cnt=cnt+1
                    pass
                if(value["Topic"] == "Spectralscan"):
                    cnt=cnt+1
                if(value["Topic"] == "Survey"):
                    cnt=cnt+1
                if(value["Topic"] == "Statistics"):
                    cnt=cnt+1
                if(value["Topic"] == "ath9kERR"):
                    cnt=cnt+1
                
    print("typecnt = " + str(typecnt))
        
def pairing(wapi, processed_data, mdb_ML, mdb_ML_pair):        
#    pairing(proc_dict, pairing_topic, found_data1, mdb_ML, mdb_ML_pair)
#    m = mongodb_api(user='ubuntu', pwd='ubuntu', database='wifi_diagnosis',collection=W_collection)    
    mdb_ML.remove(key_value = {}, justone=False)    
    mdb_ML_pair.remove(key_value = {}, justone=False)
    
    TrainData=[]
    TestData=[]
    
    for data in processed_data:
        if data["Casetype"]=="Train":
            TrainData.append(data)
        if data["Casetype"]=="Test":
            TestData.append(data)
        
    print("len(TrainData) = " + str(len(TrainData)))
    print("len(TestData) = " + str(len(TestData)))
    
    for data in TestData:
        mdb_ML.insert(data)
    
    idx=0
    for data in TrainData:
        for labeldata in wapi.label_list:
            if(labeldata["MLType"]=="Out"):
                data["AP"][labeldata["Name"]] = TestData[idx]["AP"][labeldata["Name"]]
                data["STA"][labeldata["Name"]] = TestData[idx]["STA"][labeldata["Name"]]
        idx = idx+1
        mdb_ML_pair.insert(data)
        if(idx>=len(TestData)):
            break
        
    print("idx" + str(idx))
    
    
def moving_average(ip_list, step):
    op_list=[]
    i=0
    for data in ip_list[step:]:
        op_list.append(np.mean(ip_list[i:i+step], axis=0))
        i=i+1
    return op_list    
    
def cut_ratio(ip_list, threshold):
    cnt = 0
    for data in ip_list:
        if(data > threshold):
            cnt = cnt + 1;
    return cnt / len(ip_list)

def GetDataList(dev, found_data, name, proc):
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

if __name__ == '__main__': 
    
    wapi = wifi_diag_api()
        
    date='1070208small-t1-2'
    SkipFirstN=0
    
    
    iptable = {"AP":"10.144.24.24", "STA":"10.144.24.23", 
               "10.144.24.24":"AP", "10.144.24.23":"STA"}
    
    R_coll = date + '-TestData'
    W_coll = date + '-ProcessData'    
    ML_coll = date + '-MLData'
    ML_coll_pair = date + 'MLData-Pair'    
    
    mR = mongodb_api(user='ubuntu', pwd='ubuntu', database='wifi_diagnosis',collection=R_coll)
    mW = mongodb_api(user='ubuntu', pwd='ubuntu', database='wifi_diagnosis',collection=W_coll)
    mdb_ML = mongodb_api(user='ubuntu', pwd='ubuntu', database='wifi_diagnosis',collection=ML_coll)
    mdb_ML_pair = mongodb_api(user='ubuntu', pwd='ubuntu', database='wifi_diagnosis',collection=ML_coll_pair)
    
    
#    sorted_data = Testdata_sort(mR)
#    data_process(wapi, sorted_data, mW, iptable, SkipFirstN)

    found_data1 = mW.find(key_value = {}, ftype='many')
    print("len(found_data1) = " + str(len(found_data1)))      
    
#    pairing(wapi, found_data1, mdb_ML, mdb_ML_pair)    
    
    # ======================================================= PLOT ====================================================
    APdf, STAdf = wapi.plot_all(mdb_ML)
    
    