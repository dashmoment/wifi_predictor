import sys
sys.path.append("../")
import numpy as np
import pandas as pd
import math

from mongodb import mongodb_api
from utility import io
from feature_extraction.signal_embedding import run_signal_embedding

class feature_extraction:


    def _getData(self, coll_prefix):
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
                    mW = mongodb_api.mongodb_api(user='ubuntu', pwd='ubuntu', database='wifi_diagnosis',collection=W_coll)
                    found = mW.find(key_value = {}, ftype='many')
                    io.c_print(W_coll + ' : ' + str(len(found)))
                    fdata = fdata + found
                    
        return fdata


    def _feature_extraction(self, found_data, dev, label, func, *args):
        
        if(callable(label)):
            return label(found_data, dev)
        
        raw_data = []
        for data in found_data:
    #        dev = "AP"
            if label not in data[dev]:
                continue
            if(type(data[dev][label])==list):
                raw_data = raw_data + data[dev][label]
            else:
                raw_data.append(data[dev][label])
           
        proc_result = func(raw_data, *args)
        
        return proc_result

    def label_generator(self, df):

        label = {}    
        label['delay_mean'] = df[['Time', 'Delay-mean']]
        label['delay_mean_log'] = pd.concat([df['Time'], np.log1p(df['Delay-mean'])], axis = 1)
        label['delay_max'] = df[['Time', 'Delay-mean']]
        label['delay_max_log'] = pd.concat([df['Time'], np.log1p(df['Delay-max'])], axis = 1)
        
        df.drop('Delay-mean', axis=1, inplace=True) 
        df.drop('Delay-max', axis=1, inplace=True)
        
        return label
    
    def cut_location(self, case_length, time_step = 10):
        
        # === a function to record the interval of data used to compute average ===
        # list to record output
        interval = []
        
        max_int = case_length//time_step
        for pos in np.arange(max_int):  # position
            interval.append(np.arange(pos*10, pos*10+10))
        
        if case_length % time_step != 0: # the last sample
            interval.append(np.arange(max_int*10, case_length))
        
        return interval
    
    def get_train_test_interval(self, data_length, num_interval, time_step):

        # === a function to sample training and testing interval with data length given
        maxInt = data_length//time_step - 1
        selectInt = np.random.choice(np.arange(maxInt), num_interval, replace=False)
        selectInt = np.sort(selectInt)

        testing_interval = [x for x in zip(selectInt*time_step, (selectInt+1)*time_step)]
        if selectInt[0] == 0: # deal with special case when testing interval starting with 0
            training_interval = [x for x in zip(selectInt[1:]*time_step, (selectInt[:-1]+1)*time_step)]
        else:
            training_interval = [x for x in zip(np.append(0, selectInt[:-1]+1)*time_step, selectInt*time_step)]
        training_interval.append([training_interval[-1][1], data_length])
        training_interval = [x for x in training_interval if x[1]-x[0] != 0]
    
        return training_interval, testing_interval

    def generator(self, coll_prefix, time_step = 15, num_interval = 4000, special_list = [], embed_model_path = ""):
            
            # ==== Create proper format for machine learning

            fdata = self._getData(coll_prefix) 
            devlist = ['AP', 'STA']
            labellist = ['Rcv', 'CCK_ERRORS', 'CRC-ERR', 'FCSError', 'OFDM_ERRORS', 'SS_Rssi']
            
            training_interval, testing_interval = self.get_train_test_interval(len(fdata), num_interval=num_interval, time_step=time_step)
            
            ### process training data
            #init continer for dictionary to dataFrame
            ProcMLData={}
            #ProcMLData['Time'] = []
            train_TimeRecord = {}
            train_TimeRecord['Time'] = []
            
            for dev in devlist:
                ProcMLData['Delay-mean'] = []
                ProcMLData['Delay-max'] = []
                for label in labellist:
                    ProcMLData[dev+'-'+label+'-mean'] = []
                    
                if 'SS_Subval' in special_list:
                    #SS_Subval should have 56, if length not 56--> ignore
                    #56 is number of feqency  
                    for i in range(56): 
                        ProcMLData[dev+'-SS_Subval-mean-'+str(i)] = []
            
            for interval in training_interval:
                for idx in range(interval[0], interval[1] - time_step + 1):

                    isNull = False         
                    current = list(fdata[idx : idx + time_step]) #Get current time step data

                    SS_Subval_tmp = {}
                    features_tmp = {}

                    for dev in devlist:
                        if 'SS_Subval' in special_list:            
                            mean_SS_Subval = self._feature_extraction(current, dev, 'SS_Subval', np.mean,0)
                            #SS_Subval should have 56, if length not 56--> ignore
                            if isinstance(mean_SS_Subval,  np.ndarray)  and len(mean_SS_Subval) == 56:
                                for i in range(56):
                                    SS_Subval_tmp[dev +'-SS_Subval-mean-'+str(i)] = mean_SS_Subval[i]    
                            else:          
                                isNull = True
                                                
                        for label in labellist:  
                            feature = self._feature_extraction(current, dev, label, np.mean)       
                            
                            if math.isnan(feature): 
                                isNull = True
                                break
                            else:
                                features_tmp[dev+'-'+label+'-mean'] = feature

                        if isNull: break
                        else: continue 
                    
                    if not isNull:   
                        
                        if 'SS_Subval' in special_list:
                            for k in SS_Subval_tmp:
                                ProcMLData[k].append(SS_Subval_tmp[k])
                        for k in features_tmp:
                            ProcMLData[k].append(features_tmp[k])                
                        
                        ProcMLData['Delay-mean'].append(self._feature_extraction(current, 'AP', 'Delay', np.mean))
                        ProcMLData['Delay-max'].append(self._feature_extraction(current, 'AP', 'Delay', np.max))

                        #********Check Length**********
                        stadard_len = len(ProcMLData['Delay-mean'])
                        for k in ProcMLData:
                            #if k == 'Time':
                            #    continue
                            if len(ProcMLData[k]) != stadard_len:
                                raise Exception('Length error:{},{}, {}, {}'.format(idx, k, stadard_len, len(ProcMLData[k])))
                            else:
                                continue
                    #ProcMLData['Time'].append(fdata[idx]['Time'])
                    train_TimeRecord['Time'].append(fdata[idx]['Time'])
            
            train = pd.DataFrame(ProcMLData)
            train_TimeRecord = pd.DataFrame(train_TimeRecord)
            train = pd.concat([train_TimeRecord, train], axis = 1)
            label_train = self.label_generator(train)

            #Add SS_Subval_emb: which is embeded 56 SS_Subval into N embeded signal by NN model    
            if 'SS_Subval' in special_list and embed_model_path != "":
                
                signal_embedding = run_signal_embedding.signal_embeding(embed_model_path)
                embeded_signal = signal_embedding.predict(train)
                
                #spike_cols = [col for col in MLdf.columns if 'SS_Subval' in col]
                #MLdf.drop(spike_cols, axis=1, inplace=True) 
                
                train = pd.concat([train, embeded_signal], axis=1)
                            
            
            ### process testing data
            #init continer for dictionary to dataFrame
            ProcMLData={}
            test_TimeRecord = {}
            test_TimeRecord['Time'] = []

            for dev in devlist:
                ProcMLData['Delay-mean'] = []
                ProcMLData['Delay-max'] = []
                for label in labellist:
                    ProcMLData[dev+'-'+label+'-mean'] = []
                    
                if 'SS_Subval' in special_list:
                    #SS_Subval should have 56, if length not 56--> ignore
                    #56 is number of feqency  
                    for i in range(56): 
                        ProcMLData[dev+'-SS_Subval-mean-'+str(i)] = []
            
            for interval in testing_interval:
                isNull = False
                current = list(fdata[interval[0]: interval[1]])  # Get current time step data

                SS_Subval_tmp = {}
                features_tmp = {}

                for dev in devlist:
                    if 'SS_Subval' in special_list:
                        mean_SS_Subval = self._feature_extraction(
                            current, dev, 'SS_Subval', np.mean, 0)
                        #SS_Subval should have 56, if length not 56--> ignore
                        if isinstance(mean_SS_Subval,  np.ndarray) and len(mean_SS_Subval) == 56:
                            for i in range(56):
                                SS_Subval_tmp[dev + '-SS_Subval-mean-' +
                                              str(i)] = mean_SS_Subval[i]
                        else:
                            isNull = True

                    for label in labellist:
                        feature = self._feature_extraction(current, dev, label, np.mean)

                        if math.isnan(feature):
                            isNull = True
                            break
                        else:
                            features_tmp[dev+'-'+label+'-mean'] = feature

                    if isNull:
                        break
                    else:
                        continue

                if not isNull:

                    if 'SS_Subval' in special_list:
                        for k in SS_Subval_tmp:
                            ProcMLData[k].append(SS_Subval_tmp[k])
                    for k in features_tmp:
                        ProcMLData[k].append(features_tmp[k])

                    ProcMLData['Delay-mean'].append(
                        self._feature_extraction(current, 'AP', 'Delay', np.mean))
                    ProcMLData['Delay-max'].append(self._feature_extraction(current,
                                                                            'AP', 'Delay', np.max))

                    #********Check Length**********
                    stadard_len = len(ProcMLData['Delay-mean'])
                    for k in ProcMLData:
                        #if k == 'Time':
                        #    continue
                        if len(ProcMLData[k]) != stadard_len:
                            raise Exception('Length error:{},{}, {}, {}'.format(interval[0], k, stadard_len, len(ProcMLData[k])))
                        else:
                            continue
                #ProcMLData['Time'].append(fdata[idx]['Time'])
                test_TimeRecord['Time'].append(fdata[interval[0]]['Time'])

            test = pd.DataFrame(ProcMLData)
            test_TimeRecord = pd.DataFrame(test_TimeRecord)
            test = pd.concat([test_TimeRecord, test], axis=1)
            label_test = self.label_generator(test)

            #Add SS_Subval_emb: which is embeded 56 SS_Subval into N embeded signal by NN model
            if 'SS_Subval' in special_list and embed_model_path != "":

                signal_embedding = run_signal_embedding.signal_embeding(embed_model_path)
                embeded_signal = signal_embedding.predict(train)

                #spike_cols = [col for col in MLdf.columns if 'SS_Subval' in col]
                #MLdf.drop(spike_cols, axis=1, inplace=True)

                test = pd.concat([test, embeded_signal], axis=1)


            return train, label_train, test, label_test
