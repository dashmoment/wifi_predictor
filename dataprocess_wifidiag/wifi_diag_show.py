# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 16:52:53 2018

@author: MaggieYC_Pang
"""

from mongodb_api import mongodb_api
from wifi_diag_api import wifi_diag_api
import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd

if __name__ == '__main__':    
        
    wapi = wifi_diag_api()    
            
    date='1070202small-t1'
    AVG_STEP = 10
        
    iptable = {"AP":"10.144.24.24", 
           "STA":"10.144.24.23", 
           "10.144.24.24":"AP", 
           "10.144.24.23":"STA"}
    
    ML_coll = date + '-MLData'
    ML_coll_pair = date + 'MLData-Pair'    
    
    mdb_ML = mongodb_api(user='ubuntu', pwd='ubuntu', database='wifi_diagnosis',collection=ML_coll)
    mdb_ML_pair = mongodb_api(user='ubuntu', pwd='ubuntu', database='wifi_diagnosis',collection=ML_coll_pair)    
    
    # =========================================================================================================  
                
    m = mdb_ML
    
    ML_data_df = wapi.create_df(m, step=AVG_STEP, func=np.mean, arg=None)
    class_df, fclass = wapi.classification(ML_data_df)
    