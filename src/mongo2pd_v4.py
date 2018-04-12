import numpy as np
import pandas as pd
import run_signal_embedding




def Feature_Extraction(found_data, dev, label, func, *args):
    
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


def mongo2pd(fdata, time_step, special_list = []):
    
        signal_embedding = run_signal_embedding.signal_embeding('signal_embeded.h5')
    
        
        # ==== Create proper format for machine learning
        
        ProcMLData={}
        
        devlist = ['AP', 'STA']
        labellist = ['Rcv', 'CCK_ERRORS', 'CRC-ERR', 'FCSError', 'OFDM_ERRORS', 'SS_Rssi']
        
        
        for dev in devlist:
            ProcMLData['Delay-mean'] = []
            ProcMLData['Delay-max'] = []
            for label in labellist:
                ProcMLData[dev+'-'+label+'-mean'] = []
                
            if 'SS_Subval' in special_list:
                for i in range(56):
                    ProcMLData[dev+'-SS_Subval-mean-'+str(i)] = []
                    
    #            for p in [1, 25, 50, 75, 99]:
    #                ProcMLData[dev+'-'+label+'-'+str(p)] = []
        
        for idx in range(len(fdata)-time_step):
         
            current = fdata[idx : idx + time_step]
            
            if 'SS_Subval' in special_list:
                mean_SS_Subval = []
                for dev in devlist:
                    
                    mean_SS_Subval.append(Feature_Extraction(current, dev, 'SS_Subval', np.mean,0))
                    
                if np.shape(mean_SS_Subval) != (len(devlist), 56): continue
                else:
                    for j in range(len(devlist)):
                        for i in range(56):
                            ProcMLData[devlist[j]+'-SS_Subval-mean-'+str(i)].append(mean_SS_Subval[j][i])
                
            for dev in devlist:
                                       
                for label in labellist:
                    ProcMLData[dev+'-'+label+'-mean'].append(Feature_Extraction(current, dev, label, np.mean))
            
            
            ProcMLData['Delay-mean'].append(Feature_Extraction(current, 'AP', 'Delay', np.mean))
            ProcMLData['Delay-max'].append(Feature_Extraction(current, 'AP', 'Delay', np.max))
            
        
            
        MLdf = pd.DataFrame(ProcMLData)   
        
        if 'SS_Subval' in special_list:
            embeded_signal = signal_embedding.predict(MLdf)
            
            spike_cols = [col for col in MLdf.columns if 'SS_Subval' in col]
            MLdf.drop(spike_cols, axis=1, inplace=True) 
            
            MLdf = pd.concat([MLdf, embeded_signal], axis=1)
                        
        
        return MLdf