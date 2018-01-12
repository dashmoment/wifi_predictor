#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 14:17:59 2017

@author: ubuntu
"""

import mongodb_api as db
import pandas as pd
import sys


    

def create_empty_data(key_pair, default = -500):
    
    data = {}
    
    for i in range(len(key_pair)):
        data[key_pair[i]] = default
        
    return data


def create_data(data, key_pair, default = -500):
    
    empy_data = create_empty_data(key_pair, default = default)
    for key in data:
        empy_data[key] = data[key]
    
    return empy_data



class test_create_data:
    
    def __init__(self, found_data):
        self.found_data = found_data
        self. idx = -1
        self.issucess = True
        
    def sucess_create_data(self,index):
        
        rawdata = self.found_data[index]["key"]
        test = create_data( rawdata, key_pair)
            
        for key in rawdata:
            
            if rawdata[key] != test[key]: return False
            
        
        return True
    
    def test(self):
        
       
        for idx in range(len(self.found_data)):
    
            self.issucess =  self.sucess_create_data(idx) and self.issucess 
            
        
        return self.issucess
    
  

#Init mogodb connection and find data
m = db.mongodb_api(user='ubuntu', pwd='ubuntu', collection='Fingerprint_171101')
found_data = m.find(ftype='many')

"""
#Simple function test
test = test_create_data(found_data)
print("Data test Result: ", test)     
"""
#Create data dimensions to fix input dimensions
#Create label table for one got encode

key_pair = []
label_pair = {}
label_count = 0
for i in range(len(found_data)):
    for key in found_data[i]["key"].keys():
        if key not in key_pair:
            key_pair.append(key)
    
    if found_data[i]["class"] not in label_pair:
        label_pair[found_data[i]["class"]] = label_count
        label_count += 1  

processed_data = []

for i in range(len(found_data)):
    rawdata = found_data[i]["key"]    
    singledata = create_data(rawdata, key_pair)
    singledata["label"] = label_pair[found_data[i]["class"]] 
    processed_data.append(singledata)
    
    
    sys.stdout.write("Process: {}/{}".format(i,len(found_data)) + "\r")
    sys.stdout.flush()
    
pd_data  = pd.DataFrame.from_dict(processed_data)

store = pd.HDFStore('../data/raw_data.h5')
store['raw_data'] = pd_data
            


    
    






























