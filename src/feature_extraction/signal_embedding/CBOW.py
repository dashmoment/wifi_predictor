import sys
sys.path.append("../")
sys.path.append("../../")
import warnings
warnings.filterwarnings("ignore")

import os
import matplotlib.pyplot as plt

# plt.switch_backend('agg')

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm, skew
from tqdm import tqdm
import pickle
import random as rnd

import keras
from keras.layers.embeddings import Embedding
import keras.backend as K
from keras.layers import Input, Dense, Activation, BatchNormalization, Reshape, merge
from keras.models import Model
from keras.models import model_from_json

import label_generator as label_gen
import label_generator_rahul as label_gen_r
from utility import io, keras_event_callBack, calculate_classWeight
import feature_extraction
import train_test_config as conf

def batch_generator(dataSet):
    
    while True:   
        pitch_idx = rnd.randint(1,len(dataSet)-1)
        #print("Pitch idx: ", pitch_idx)
        yield dataSet[pitch_idx]

def negtive_sampling(batchSize, data, label_oh):
        #Generate sample
        while True:
            context_sample = data.sample(1)
            label = label_oh.iloc[context_sample.index].idxmax(axis=1).iloc[0]
            
            train_sample_pool =  data[~data.index.isin(context_sample.index)]
            pos_sample = train_sample_pool[label_oh.idxmax(axis=1)==label].sample(batchSize//2) 
            neg_sample = train_sample_pool[label_oh.idxmax(axis=1)!=label].sample(batchSize//2)
            
            context_sample = context_sample.append([context_sample]*(batchSize-1))
            target_sample = pd.concat([pos_sample, neg_sample])
            train_label = pd.DataFrame([1 for i in range(batchSize//2)] + [0 for _ in range(batchSize//2, batchSize)])
            
            yield ([context_sample, target_sample], train_label)
            

if __name__ == '__main__': 
    
    
    #Check dataset is generated, if not then generate it.
    
    filePath = '../../../data/skip_gram_batchData.pkl'
    
    if os.path.isfile(filePath):
        with open(filePath, 'rb') as input:
            dataSet = pickle.load(input)
    
    else:
        fext = feature_extraction.feature_extraction()
        #Load train and test configuration
        config = conf.train_test_config('Read_Collection_train_c1', 'Read_Collection_test_ct')
            
        #Generator train & test data by configuration 
        train, label_train = fext.generator(config.train, time_step=15,  special_list = ['SS_Subval'])   
        #Extract subval and normalization
        train_AP_SS =  train[[cols for cols in train.columns if 'AP-SS_Subval' in cols]]
        train_AP_SS = (train_AP_SS - train_AP_SS.mean())/train_AP_SS.std()
        train_STA_SS =  train[[cols for cols in train.columns if 'STA-SS_Subval' in cols]]
        train_STA_SS = (train_STA_SS - train_STA_SS.mean())/train_STA_SS.std()    
        label_train_dm =  label_gen_r.TransferToOneHotClass(label_train['delay_mean'])
        
        dataSet = []
        gen_batch = negtive_sampling(64, train_AP_SS, label_train_dm)
        for i in tqdm(range(100000)):
            batch_data = next(gen_batch)
            dataSet.append(batch_data)
        
        with open(filePath, 'wb') as output:
            pickle.dump(dataSet, output, pickle.HIGHEST_PROTOCOL)
    
    '''
    fext = feature_extraction.feature_extraction()
    #Load train and test configuration
    config = conf.train_test_config('Read_Collection_train_c1', 'Read_Collection_test_ct')
            
    #Generator train & test data by configuration 
    train, label_train = fext.generator(config.train, time_step=15,  special_list = ['SS_Subval'])   
    #Extract subval and normalization
    train_AP_SS =  train[[cols for cols in train.columns if 'AP-SS_Subval' in cols]]
    train_AP_SS = (train_AP_SS - train_AP_SS.mean())/train_AP_SS.std()
    train_STA_SS =  train[[cols for cols in train.columns if 'STA-SS_Subval' in cols]]
    train_STA_SS = (train_STA_SS - train_STA_SS.mean())/train_STA_SS.std()    
    label_train_dm =  label_gen_r.TransferToOneHotClass(label_train['delay_mean'])
    '''
    
    #Train model
    batchSize = 128
    epochs = 5000
    savePath_root =  '../../../trained_model/skip_gram_enc/'
    
    if os.path.exists(savePath_root):
        
        print(" Load Trained Model")
        json_file = open(os.path.join(savePath_root, 'graph.json'), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights(os.path.join(savePath_root, 'weight.h5'))
          
    else:
        print("Model not exist. Build new model")
        os.mkdir(savePath_root)
         #Build model
        vector_dim = 28
        input_target = Input((56,))
        input_context = Input((56,))
        embedding = Dense(vector_dim, activation='sigmoid')
        #embedding = Embedding(56, vector_dim, input_length=1, name='embedding')
        target = embedding(input_target)
        target = Reshape((vector_dim, 1))(target)
        context = embedding(input_context)
        context = Reshape((vector_dim, 1))(context)
        
        # now perform the dot product operation to get a similarity measure
        dot_product = merge([target, context], mode='dot', dot_axes=1)
        dot_product = Reshape((1,))(dot_product)
        # add the sigmoid output layer
        output = Dense(1, activation='sigmoid')(dot_product)
        # create the primary training model
        model = Model(input=[input_target, input_context], output=output)
    
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])     
    
    saveModel_cb = keras_event_callBack.saveModel_Callback(
                                                            10,
                                                            model,
                                                            os.path.join(savePath_root, 'graph.json'),
                                                            os.path.join(savePath_root, 'weight.h5')
                                                            )
    tensorBoard_cb = keras_event_callBack.tensorBoard_Callback(log_dir=os.path.join(savePath_root, 'logs'))
   
    history = model.fit_generator(
                                    batch_generator(dataSet),
                                    validation_data = batch_generator(dataSet),
                                    steps_per_epoch = 10000,
                                    epochs =epochs, 
                                    validation_steps=10000,
                                    callbacks = [saveModel_cb, tensorBoard_cb]
                                    )
    
#    history = model.fit_generator(
#                                    negtive_sampling(128, train_AP_SS, label_train_dm),
#                                    validation_data = negtive_sampling(128, train_AP_SS, label_train_dm),
#                                    steps_per_epoch = 10000,
#                                    epochs =epochs, 
#                                    validation_steps=10000,
#                                    callbacks = [saveModel_cb, tensorBoard_cb]
#                                    )
    

    #predict = model.predict(next(batch_generator(dataSet))[0])
   
    
    
    
    
    
    
    
    
    
    
    