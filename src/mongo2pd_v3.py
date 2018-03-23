import numpy as np

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
       
#    print("func:" + str(func))
#    print("kwargs:" + str(kwargs))
    
    # A Value
    proc_result = func(raw_data, *args)
    
    return proc_result
 


def mongo2pd(fdata, time_step):
    
        stepsize = 10
        # ==== Create proper format for machine learning
        
        ProcMLData={}
        
        devlist = ['AP', 'STA']
        labellist = ['Rcv', 'CCK_ERRORS', 'CRC-ERR', 'FCSError', 'OFDM_ERRORS', 'SS_Rssi']
        
        for dev in devlist:
            ProcMLData['Delay-mean'] = []
            ProcMLData['Delay-max'] = []
            for label in labellist:
                ProcMLData[dev+'-'+label+'-mean'] = []
    #            for p in [1, 25, 50, 75, 99]:
    #                ProcMLData[dev+'-'+label+'-'+str(p)] = []
        
        for idx in range(len(fdata)-stepsize):
            current = fdata[idx : idx + stepsize]
            ProcMLData['Delay-mean'].append(Feature_Extraction(current, 'AP', 'Delay', np.mean))
            ProcMLData['Delay-max'].append(Feature_Extraction(current, 'AP', 'Delay', np.max))
            for dev in devlist:
                for label in labellist:
                    ProcMLData[dev+'-'+label+'-mean'].append(Feature_Extraction(current, dev, label, np.mean))
                    
        
        return ProcMLData