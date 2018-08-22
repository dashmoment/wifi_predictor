import sys
sys.path.append("../")
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tensorflow.python.framework import ops
import keras
import keras.backend as K
from keras.layers import Input, Dense, Activation, BatchNormalization
from keras.models import Model
from sklearn.feature_selection import SelectKBest, f_regression
import pickle
from sklearn.manifold import TSNE

from utility import save_load_Keras, plot_tSNE, keras_event_callBack
import train_test_config as conf
from feature_extraction import label_generator_rahul as label_gen_r
from feature_extraction import feature_engineering
from feature_extraction import feature_extraction
from model.nn_model_rahul_keras import nn_model

def calculate_classWeight(label):    
    
    class_num = label.sum()
    _weight = len(label_train_dm)/(class_num + 1e-4)
    classWeight = {}
    for idx, value in enumerate(_weight):
        classWeight[idx] = value
    
    return classWeight

if __name__ == '__main__':
    
    #Load train and test configuration
    config = conf.train_test_config('Read_Collection_train_c1', 'Read_Collection_test_ct')
        
    #Generator train & test data by configuration 
    fext = feature_extraction.feature_extraction()
    train, label_train = fext.generator(config.train, time_step=15,  special_list = ['SS_Subval'])
       
    
    #Extract subval and normalization
    train_AP_SS =  train[[cols for cols in train.columns if 'AP-SS_Subval' in cols]]
    train_AP_SS = (train_AP_SS - train_AP_SS.mean())/train_AP_SS.std()
    train_STA_SS =  train[[cols for cols in train.columns if 'STA-SS_Subval' in cols]]
    train_STA_SS = (train_STA_SS - train_STA_SS.mean())/train_STA_SS.std()    
    label_train_dm =  label_gen_r.TransferToOneHotClass(label_train['delay_mean'])
    
    #Reload Encoder model and gen encoded features
    encoder_model = save_load_Keras.load_model('../../trained_model/nn_encoder/nn_encoder.json',
                                               '../../trained_model/nn_encoder/nn_encoder.h5')  
    encoder_model.summary() 
    encoder_model = Model(inputs=encoder_model.input,outputs=encoder_model.get_layer('activation_10').output)
    encoder_model.summary()
    
    encoded_feature = encoder_model.predict(train_AP_SS)
    
    #Plot encoded feature - class
#    tsne_plotter = plot_tSNE.plot_tSNE(2000, encoded_feature,label_train_dm)
#    tsne_plotter.plot()

        
    #Test Accuracy
#    predict = np.argmax(encoded_feature, axis=1)
#    label = np.argmax(np.array(label_train_dm), axis=1)
#    accuracy = np.mean(np.equal(label, predict))
    
    
    #Load Original feature
    
    with open('../../data/raw_data_wo_time_sub.pkl', 'rb') as input:

        train_leagacy = pickle.load(input)
        label_train = pickle.load(input)
        test_raw = pickle.load(input)
        label_test_raw = pickle.load(input)
        
    train_leagacy = feature_engineering.binding(train_leagacy)
    label_train_dm =  label_gen_r.TransferToOneHotClass(label_train['delay_mean'])
    anova_filter = SelectKBest(f_regression, k=10).fit(train_leagacy, label_train['delay_mean'])
    mask = anova_filter.get_support(indices=True)
    print('Selected features: {}'.format(train_leagacy.columns[mask]))
    
   
    #train, label_train, test, label_test = label_gen_r.RandomSample(train, label_train['delay_mean'], fraction=1.0)
    
    #Plot Legacy feature - class
#    train_selected_f = train_leagacy[train_leagacy.columns[mask]]
#    tsne_plotter = plot_tSNE.plot_tSNE(2000, train_selected_f,label_train_dm, plot2d=True)
#    tsne_plotter.plot()
    
    
    #Concat encode features + original features
    encoded_feature_df_colsName = ['AP_SS_encode_'+str(i) for i in range(32)] 
    encoded_feature_df = pd.DataFrame(encoded_feature, columns=encoded_feature_df_colsName, dtype=np.float32) 
    #train = pd.concat([train_leagacy, encoded_feature_df], axis=1)
    train = train_leagacy
    anova_filter = SelectKBest(f_regression, k=10).fit(train, label_train['delay_mean'])
    mask = anova_filter.get_support(indices=True)
    print('Selected features: {}'.format(train.columns[mask]))
    
    train = train.sample(frac=1.0)    
    label_train_shuffle = label_train['delay_mean'].iloc[train.index]
   
    train_set = [train.iloc[:, mask],
                 label_gen_r.TransferToOneHotClass(label_train_shuffle)]
   
    # normalization
    train_set[0] = (train_set[0] - train_set[0].mean())/train_set[0].std()
    
    #ops.reset_default_graph()
    
    rahul_model = nn_model(train_set[0].shape[1:])
    rahul_model.summary()
    adam = keras.optimizers.Adam(lr=0.0005, epsilon=1e-8)
    rahul_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    
    savePath_root =  '../../trained_model/nn_model_rahul_wo_enc/'
    
    saveModel_cb = keras_event_callBack.saveModel_Callback(
                                                            10,
                                                            rahul_model,
                                                            os.path.join(savePath_root, 'graph.json'),
                                                            os.path.join(savePath_root, 'weight.h5')
                                                            )
    tensorBoard_cb = keras_event_callBack.tensorBoard_Callback(log_dir=os.path.join(savePath_root, 'logs'))
    
    history = rahul_model.fit(train_set[0], train_set[1],
                       epochs=1000,
                       validation_split=0.2,
                       batch_size=64,
                       class_weight=calculate_classWeight(train_set[1]),
                       callbacks = [saveModel_cb, tensorBoard_cb] 
                       )
    
    
    
    