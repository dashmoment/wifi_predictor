import sys
sys.path.append("../")
from mongodb import mongodb_api
from utility import io
import numpy as np
import pandas as pd
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
        label['delay_mean'] = df['Delay-mean']
        label['delay_mean_log'] = np.log1p(df['Delay-mean'])
        label['delay_max'] = df['Delay-max']
        label['delay_max_log'] = np.log1p(df['Delay-max'])
          
        df.drop('Delay-mean', axis=1, inplace=True) 
        df.drop('Delay-max', axis=1, inplace=True)
        
        return label

    def generator(self, coll_prefix, time_step, special_list = [], embed_model_path = ""):
            
            # ==== Create proper format for machine learning

            fdata = self._getData(coll_prefix) 
            devlist = ['AP', 'STA']
            labellist = ['Rcv', 'CCK_ERRORS', 'CRC-ERR', 'FCSError', 'OFDM_ERRORS', 'SS_Rssi']

            ProcMLData={}
            
            #init continer for dictionary to dataFrame
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
            
            for idx in range(len(fdata)-time_step):
             
                current = fdata[idx : idx + time_step] #Get current time step data
                
                if 'SS_Subval' in special_list:
                    mean_SS_Subval = []
                    for dev in devlist:
                        
                        mean_SS_Subval.append(self._feature_extraction(current, dev, 'SS_Subval', np.mean,0))
                    
                    #SS_Subval should have 56, if length not 56--> ignore
                    if np.shape(mean_SS_Subval) != (len(devlist), 56): continue
                    else:
                        for j in range(len(devlist)):
                            for i in range(56):
                                ProcMLData[devlist[j]+'-SS_Subval-mean-'+str(i)].append(mean_SS_Subval[j][i])
                    
                for dev in devlist:
                                           
                    for label in labellist:
                        ProcMLData[dev+'-'+label+'-mean'].append(self._feature_extraction(current, dev, label, np.mean))
                
                
                ProcMLData['Delay-mean'].append(self._feature_extraction(current, 'AP', 'Delay', np.mean))
                ProcMLData['Delay-max'].append(self._feature_extraction(current, 'AP', 'Delay', np.max))
                
            df = pd.DataFrame(ProcMLData)  
            label = self.label_generator(df)

            #Add SS_Subval_emb: which is embeded 56 SS_Subval into N embeded signal by NN model    
            if 'SS_Subval' in special_list and embed_model_path != "":
                
                signal_embedding = run_signal_embedding.signal_embeding(embed_model_path)
                embeded_signal = signal_embedding.predict(df)
                
                #spike_cols = [col for col in MLdf.columns if 'SS_Subval' in col]
                #MLdf.drop(spike_cols, axis=1, inplace=True) 
                
                df = pd.concat([df, embeded_signal], axis=1)
                            
            
            return df, label