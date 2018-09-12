import sys
sys.path.append("../")
sys.path.append("../../")
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
import pickle
import importlib
from datetime import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import random as rnd

from sklearn import metrics
from sklearn import linear_model
from sklearn import svm
from sklearn.svm import SVR
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

from tensorflow.python.framework import ops
from scipy import stats
from scipy.stats import norm, skew

import keras
import keras.backend as K
from keras.layers import Input, Dense, Activation, BatchNormalization
from keras.models import Model

import label_generator as label_gen
import label_generator_rahul as label_gen_r
from model.nn_model_rahul_keras import nn_model
from utility import io, keras_event_callBack
import feature_extraction
import train_test_config as conf

def generate_randomMask(length, amount):
    
    assert length > amount, "Length should larger than amount"
    randomMask = np.zeros(length, dtype=bool)
    count = 0
    while count < amount:
        rnd_num = rnd.randint(0, length)
        if  randomMask[rnd_num] !=True: 
            randomMask[rnd_num] = True
            count+=1
        else: continue
    
    return randomMask
    


def build_encoder(inputShpae, initializer='he_uniform'):
    
    layerIndex = 0
    Nfilters = 32
    
    x_input = x = Input(inputShpae) 
    for i in range(4):

        x = Dense(Nfilters, kernel_initializer=initializer, name='fc'+str(layerIndex))(x)
        x = Activation('relu')(x)    
        x = BatchNormalization(name='bn'+str(layerIndex))(x)
        layerIndex+=1
        
        

    prediction = Dense(4, activation='softmax', kernel_initializer=initializer, name='output')(x)
    
    model = Model(inputs=x_input, outputs=prediction, name='DNN_encoder')
    model.summary()
    return model
 

def calculate_classWeight(label):    
    
    class_num = label.sum()
    _weight = len(label_train_dm)/(class_num + 1e-4)
    classWeight = {}
    for idx, value in enumerate(_weight):
        classWeight[idx] = value
    
    return classWeight
    
ifShowPlot = True

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
    
    
    #Display label categorical distribution
    if ifShowPlot:
        label_argmax = label_train_dm.idxmax(axis=1)
        sns.distplot(label_argmax).set_title('Label Class Hist')
        
    train_data = train_AP_SS[:int(0.8*len(train_AP_SS))]
    train_label = label_train_dm[:int(0.8*len(label_train_dm))]
    
    test_data = train_AP_SS[int(0.8*len(train_AP_SS)):]
    test_label = label_train_dm[int(0.8*len(label_train_dm)):]
        
    
    #build model    
    model_config = {
                        'batch_size': 64,
                        'validation_step': 10,
                        'epochs': 500
                    }
    
    model = build_encoder((56,))
    adam = keras.optimizers.Adam(lr=1e-4, epsilon=1e-8)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    
    saveModel_cb = keras_event_callBack.saveModel_Callback(
                                                            10,
                                                            model,
                                                            '../../trained_model/nn_encoder/nn_encoder.json',
                                                            '../../trained_model/nn_encoder/nn_encoder.h5'
                                                            )
    tensorBoard_cb = keras_event_callBack.tensorBoard_Callback(log_dir='../../trained_model/nn_encoder/logs')
    
    history = model.fit(
                        train_data,
                        train_label,
                        epochs=model_config['epochs'],
                        validation_data = (test_data, test_label),
                        class_weight=calculate_classWeight(label_train_dm),
                        callbacks = [saveModel_cb, tensorBoard_cb]                     
                        )
   
    
    