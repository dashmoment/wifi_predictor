#Train and test from different set
#Add singal value

import sys
sys.path.append("../")

from mongodb_api import mongodb_api
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew

from sklearn import metrics

from utility import c_save_image, c_print

#Manual Split training and test set

def Read_Collection_train():
    coll_prefix={}
    coll_prefix["Exception"] = []
    BigRun = { "traffic":["-Two5M","-Two10M","-Two15M","-Two20M"], 
            "ID":["-1", "-2","-3","-4","-5"]}    
    BigBigRun = { "traffic":["-Two5M","-Two10M","-Two15M","-Two20M","-Two25M","-Two30M"], 
            "ID":["-1", "-2","-3","-4","-5"]} 
    
    testrun = { "traffic":["-Two5M"], 
            "ID":["-1"]} 
    #coll_prefix["1070323-C2-L1is20"] = testrun
    
    coll_prefix["1070323-C2-L1is20"] = BigRun
    
    coll_prefix["1070322-C2-L1is30"] = BigRun

    
    coll_prefix["1070320-C2-L1is20"] = BigRun
    
    
    coll_prefix["1070320-C2-L1is30"] = BigRun
    coll_prefix["1070319-C-L1is10"] = BigRun
    
    coll_prefix["1070317-C-L1is0"] = BigRun
    coll_prefix["Exception"].append("1070317-C-L1is0-Two15M-5")
    coll_prefix["Exception"].append("1070317-C-L1is0-Two20M-5")    
    coll_prefix["Exception"].append("1070317-C-L1is0-Two5M-1")  
    
    coll_prefix["1070316-L3is40-L4is25"] = BigRun    
    
    coll_prefix["1070315-L3is40-L4is40"] = BigRun
    coll_prefix["Exception"].append("1070315-L3is40-L4is40-Two10M-3")
    
    
    coll_prefix["1070314-L3is10-L4is40"] = BigRun
    coll_prefix["Exception"].append("1070314-L3is10-L4is40-Two20M-1")
    coll_prefix["Exception"].append("1070314-L3is10-L4is40-Two10M-5")
        
    coll_prefix["1070314-L3is10-L4is25"] = BigRun
    coll_prefix["Exception"].append("1070314-L3is10-L4is25-Two5M-1")
    
    coll_prefix["1070313-L3is25-L4is25"] = BigRun
    
    coll_prefix["1070312-L3is25-L4is40"] = BigBigRun    
    coll_prefix["1070307-bigrun-L3is10"] = BigBigRun      
    coll_prefix["1070308-bigrun-L3is25"] = BigBigRun    
    coll_prefix["1070309-bigrun-L3is40"] = BigBigRun
    
  
   
    coll_prefix["1070222-clear"]={
            "ID":["","-2","-3","-4","-5"]
            }
        
    coll_prefix["1070223"]={
            "traffic":["-one10M","-one20M"], 
            "ID":["","-2","-3"]
            }    
    coll_prefix["1070227"]={
            "traffic":["-two10M","-two10Mt2"], 
            "ID":["","-2","-3"]
            }    
    coll_prefix["1070301"]={
            "traffic":["-two20M","-two20Mt2"], 
            "ID":["","-2","-3"]
            }    
    coll_prefix["1070302"]={
            "traffic":["-two20M","-two20M-L3is30"], 
            "ID":["","-2","-3"]
            }    
    coll_prefix["1070305"]={
            "traffic":["-two10M-L3is40","-two20M-L3is40"], 
            "ID":["","-2","-3"]
            }        
    
    coll_prefix["1070306"]={
            "traffic":["-two20M-L3is50","-two20M-L3isInf"], 
            "ID":["","-2","-3"]
            }
    
    
    return coll_prefix


def Read_Collection_test():
    coll_prefix={}
    coll_prefix["Exception"] = []
    BigRun = { "traffic":["-Two5M","-Two10M","-Two15M","-Two20M"], 
            "ID":["-1", "-2","-3","-4","-5"]}    
    BigBigRun = { "traffic":["-Two5M","-Two10M","-Two15M","-Two20M","-Two25M","-Two30M"], 
            "ID":["-1", "-2","-3","-4","-5"]} 
    
    NoneTraf = { "traffic":["-None"], "ID":["-1", "-2","-3","-4","-5","-6","-7","-8","-9","-10"]}    
    #coll_prefix["1070326-Free-NoInterfere"] = NoneTraf
    coll_prefix["1070328-Office"] = BigRun
    
    return coll_prefix

def GetData(coll_prefix):
    fdata = []
    for date, value in coll_prefix.items():
        if("traffic" in value):
            traflist = value["traffic"]
        else:
            traflist =[""]
        
        if("ID" in value):
            IDlist = value["ID"]
        else:
            IDlist = [""]
           
        for t in traflist:
            for i in IDlist:
                W_coll = date + t + i 
                if(W_coll in coll_prefix["Exception"]):
                    continue
                W_coll = W_coll + '-ProcessData'
                mW = mongodb_api(user='ubuntu', pwd='ubuntu', database='wifi_diagnosis',collection=W_coll)
                found = mW.find(key_value = {}, ftype='many')
                c_print(W_coll + ' : ' + str(len(found)))
                fdata = fdata + found
                
    return fdata


def label_generator(df):
    
    y_train_mean = df['Delay-mean']
    y_train_logmean = df['Delay-mean_log']
    y_train_max = df['Delay-max']
    y_train_logmax = df['Delay-max_log']
      
    df.drop('Delay-mean', axis=1, inplace=True) 
    df.drop('Delay-mean_log', axis=1, inplace=True)
    df.drop('Delay-max', axis=1, inplace=True)
    df.drop('Delay-max_log', axis=1, inplace=True)
    
    return y_train_mean, y_train_logmean, y_train_max, y_train_logmax
    

def transform_delay2category(delay):
    
    delay_cat = pd.cut(
                        delay.stack(),
                        [-np.inf,5,10,20, np.inf],
                        labels = [0,1,2,3]
                        ).reset_index(drop=True)
    
    return delay_cat


    

if __name__ == '__main__': 
    

    coll_prefix = Read_Collection_train()    
    fdata_train = GetData(coll_prefix)  
    import mongo2pd_v3 as mpd
    #ProcMLData = mpd.mongo2pd(fdata_train, time_step=15, special_list = ['SS_Subval'])
    ProcMLData = mpd.mongo2pd(fdata_train, time_step=15)
    coll_prefix = Read_Collection_test()    
    fdata_test = GetData(coll_prefix)  
    ProcMLData_test = mpd.mongo2pd(fdata_test, time_step=15)
    #ProcMLData_test = mpd.mongo2pd(fdata_test, time_step=15, special_list = ['SS_Subval'])

    c_print('Finish loading data')

    MLdf = pd.DataFrame(ProcMLData)
    MLdf_test = pd.DataFrame(ProcMLData_test)
    
    
    import delay_analyzier as delay_a
    delay_a.delay_analysis(MLdf)
    delay_a.delay_analysis(MLdf_test)
    
    '''
    Prepare training set and validation set    
    '''
    y_train_mean, y_train_logmean, y_train_max, y_train_logmax = label_generator(MLdf)    
    y_test_mean, y_test_logmean, y_test_max, y_test_logmax = label_generator(MLdf_test)    
    
    train = MLdf
    valid = MLdf_test
    y_train = y_train_logmean
    y_valid = y_test_logmean
    
    #y_train = y_train_mean
    #y_valid = y_test_mean
     
 
    '''
    Regression
    '''
    import model_conf1 
    from xgboost import plot_importance
    from sklearn.metrics import confusion_matrix
    
    c_print('Start regressor training')

    m = model_conf1.models(train, y_train, train, 5)
    m.model_config1()
    test_model = m.model_xgb
 
    test_model.fit(train, y_train)
    pred = test_model.predict(valid)
    c_print('Finish regressor training')    
    '''
    Evaluation
    '''
    rmse = m.rmsle(np.expm1(y_valid), np.expm1(pred))
    c_print('RMSE:{}'.format(rmse))
    
    #plt.figure()
    #plt.scatter(y_valid, pred)
    #plt.show()
    y_valid = pd.DataFrame(np.expm1(y_valid))
    y_pred = pd.DataFrame(np.expm1(pred))
    
    #y_valid = pd.DataFrame(y_valid)
    #y_pred = pd.DataFrame(pred)
    
    y_valid_cat = transform_delay2category(y_valid)
    y_pred_cat = transform_delay2category(y_pred)
       
    c_print('Recall reg_clf:{}'.format(metrics.recall_score(y_valid_cat, y_pred_cat, average = 'micro')))
    #c_print('Recall-2:',sum(y_valid_cat == y_pred_cat)/len(y_valid_cat))
    c_matrix_rclf = confusion_matrix(y_valid_cat, y_pred_cat)

    c_print(c_matrix_rclf)
    
    c_save_image(plt.plot, y_valid, filename='tvalid.png')
    c_save_image(plt.plot, y_pred, filename='tpred.png')
    c_save_image(plt.scatter, y_valid, y_pred, filename='tvalid_pred.png')
    
    importance=pd.Series(test_model.feature_importances_,index=train.columns).sort_values(ascending=False)
    importance.to_csv('importance.csv')

  
    
    ''' 
    Test classfier
    '''
    c_print('Start classfier training')
    import xgboost as xgb
    model = xgb.XGBClassifier(max_depth=3, learning_rate=0.05 ,n_estimators=2000, silent=True)
    
    y_train_clf = pd.DataFrame(np.expm1(y_train))
    y_train_clf = transform_delay2category(np.expm1(y_train_clf))
    model.fit(train, y_train_clf)
    y_pred_clf = model.predict(valid)
    
    c_print('Recall clf: {}'.format(metrics.recall_score(y_valid_cat, y_pred_clf, average = 'micro')))
    
    from sklearn.metrics import confusion_matrix
    xg_train_c_matrix = confusion_matrix(y_valid_cat, y_pred_clf)
    #plt.imshow(xg_train_c_matrix)
    #plt.show()
    
    c_matrix_clf = confusion_matrix(y_valid_cat, y_pred_clf)
    #plt.figure()
    #plot_importance(model)
    #plt.show()

    c_print('Finish classfier training') 

