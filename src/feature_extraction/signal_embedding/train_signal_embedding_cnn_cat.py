import keras as k
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, Conv1D, MaxPooling2D, MaxPooling1D
import numpy as np
import pandas as pd
from mongodb_api import mongodb_api

from utility import c_save_image, c_print
import train_test_config as conf

Read_Collection_train = conf.Read_Collection_train_c2


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

def mongo2pd(fdata, time_step,  special_list = []):
    
        
        # ==== Create proper format for machine learning
        
        ProcMLData={}
        
        devlist = ['AP', 'STA']
        labellist = []
        
        
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
                    
                    mean_SS_Subval.append(Feature_Extraction(current, dev, 'SS_Subval', np.mean, 0))
                    
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
            
               
                        
        
        return ProcMLData
    
    
def transform_delay2category(delay):
    
        delay_cat = pd.cut(
                            delay.stack(),
                            [0,5,10,20, np.inf],
                            labels = [0,1,2,3]
                            ).reset_index(drop=True)
        
        return delay_cat


if __name__ == '__main__': 
    
    model_path = '../model/signal_embeded_cnncat.h5'
    

    coll_prefix = Read_Collection_train()    
    fdata_train = GetData(coll_prefix)  
    ProcMLData = mongo2pd(fdata_train, time_step=15, special_list = ['SS_Subval'])
    MLdf = pd.DataFrame(ProcMLData)
    
    import delay_analyzier as delay_a
    delay_a.delay_analysis(MLdf)
    
    y_train_mean, y_train_logmean, y_train_max, y_train_logmax = label_generator(MLdf)
    
    X_train_r = np.zeros((len(MLdf), 112,1))
    X_train_r[:, :,0] = MLdf.values[:,:]
    X_train_r = np.reshape(X_train_r, (-1,56,2,1))
    
    
    y_train_logmean = pd.get_dummies(transform_delay2category(pd.DataFrame(y_train_mean)))
    
    input_shape = (56,2,1)
    model = k.Sequential()
    model.add(Conv2D(8, kernel_size=(3,1), activation='relu', input_shape = input_shape, padding='same'))
    model.add(Conv2D(8, kernel_size=(3,1), activation='relu',  padding='same'))
    model.add(MaxPooling2D((2,1)))
    model.add(Conv2D(16, kernel_size=(3,1), activation='relu',  padding='same'))
    model.add(MaxPooling2D((2,1)))
    model.add(Conv2D(16, kernel_size=(3,1), activation='relu',  padding='same'))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4,activation='softmax'))
    
    #model = k.models.load_model(model_path)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  
    history = model.fit(X_train_r, y_train_logmean, epochs=2000, batch_size=128)
    

    
    model.save(model_path)
    model = k.models.load_model(model_path)
    from keras import backend as kb
    get_embeded_layer_output = kb.function([model.layers[0].input], [model.layers[-3].output])
    embeded = get_embeded_layer_output([X_train_r])[0]
    
    predict = model.predict(X_train_r)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
