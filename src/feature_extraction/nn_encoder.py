import sys
sys.path.append("../")
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
import pickle
import importlib
from datetime import datetime
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn import metrics
from sklearn import linear_model
from sklearn import svm
from sklearn.svm import SVR
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import make_pipeline

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
from utility import io
import feature_extraction
import train_test_config as conf


def build_encoder(inputShpae, initializer='he_uniform'):
    
    layerIndex = 0
    Nfilters = 64
    
    x_input = x = Input(inputShpae) 
    for i in range(2):

        x = Dense(Nfilters, kernel_initializer=initializer, name='fc'+str(layerIndex))(x)
        x = Activation('relu')(x)    
        x = BatchNormalization(name='bn'+str(layerIndex))(x)
        layerIndex+=1
        if i%2==0 : Nfilters *= 2
        

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
    config = conf.train_test_config('Read_Collection_train_c1', 'Read_Collection_test_c1')
        
    #Generator train & test data by configuration 
    train, label_train = fext.generator(config.train, time_step=15,  special_list = ['SS_Subval'])
    test, label_test = fext.generator(config.test, time_step=15,  special_list = ['SS_Subval'])
    
    #Extract subval
    train_AP_SS =  train[[cols for cols in train.columns if 'AP-SS_Subval' in cols]]
    train_STA_SS =  train[[cols for cols in train.columns if 'STA-SS_Subval' in cols]]
    test_AP_SS =  test[[cols for cols in test.columns if 'AP-SS_Subval' in cols]]
    test_STA_SS =  test[[cols for cols in test.columns if 'STA-SS_Subval' in cols]]
    
    label_train_dm =  label_gen_r.TransferToOneHotClass(label_train['delay_mean'])
    label_test_dm =  label_gen_r.TransferToOneHotClass(label_test['delay_mean'])
    
    #Display label categorical distribution
    if ifShowPlot:
        label_argmax = label_train_dm.idxmax(axis=1)
        sns.distplot(label_argmax).set_title('Label Class Hist')
        label_argmax = label_test_dm.idxmax(axis=1)
        sns.distplot(label_argmax).set_title('Label Class Hist')
    
    #build model    
    model_config = {
                        'batch_size': 64,
                        'validation_step': 10,
                        'epochs': 500
                    }
    
    model = build_encoder((56,))
    adam = keras.optimizers.Adam(lr=1e-4, epsilon=1e-8)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    
    history = model.fit(
                        train_AP_SS,
                        label_train_dm,
                        epochs=model_config['epochs'],
                        steps_per_epoch = len(train_AP_SS)//model_config['batch_size'],
                        validation_data = (test_AP_SS, label_test_dm),
                        validation_steps = model_config['validation_step']*len(train_AP_SS)//model_config['batch_size'],
                        class_weight=calculate_classWeight(label_train_dm)
                        )
   
    
    