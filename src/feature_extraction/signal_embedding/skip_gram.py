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

import keras
from keras.layers.embeddings import Embedding
import keras.backend as K
from keras.layers import Input, Dense, Activation, BatchNormalization, Reshape, merge
from keras.models import Model

import label_generator as label_gen
import label_generator_rahul as label_gen_r
from utility import io, keras_event_callBack, calculate_classWeight
import feature_extraction
import train_test_config as conf

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
            
def negtive_sampling_validation(batchSize, data, label_oh):
        #Generate sample
        while True:
            context_sample = data.sample(1)
            label = label_oh.iloc[context_sample.index].idxmax(axis=1).iloc[0]
            
            train_sample_pool =  data[~data.index.isin(context_sample.index)]
            pos_sample = train_sample_pool[label_oh.idxmax(axis=1)==label].sample(batchSize) 
           
            
            context_sample = context_sample.append([context_sample]*(batchSize-1))
            target_sample =pos_sample
            train_label = pd.DataFrame([1 for i in range(batchSize)])
            
            yield ([context_sample, target_sample], train_label)


if __name__ == '__main__': 
    
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
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc'])
    
    model.summary()
    
    #Train model
    batchSize = 128
    epochs = 100
    savePath_root =  '../../trained_model/skip_gram_enc/'
    saveModel_cb = keras_event_callBack.saveModel_Callback(
                                                            10,
                                                            model,
                                                            os.path.join(savePath_root, 'graph.json'),
                                                            os.path.join(savePath_root, 'weight.h5')
                                                            )
    tensorBoard_cb = keras_event_callBack.tensorBoard_Callback(log_dir=os.path.join(savePath_root, 'logs'))
   
    
    history = model.fit_generator(
                                    negtive_sampling(batchSize, train_AP_SS, label_train_dm),
                                    validation_data = negtive_sampling_validation(batchSize, train_AP_SS, label_train_dm),
                                    samples_per_epoch= len(train_AP_SS)//batchSize,
                                    validation_steps = len(train_AP_SS)//batchSize,
                                    nb_epoch=epochs, 
                                    callbacks = [saveModel_cb, tensorBoard_cb]
                                    )
    

        
   
    
    
    
    
    
    
    
    
    
    
    