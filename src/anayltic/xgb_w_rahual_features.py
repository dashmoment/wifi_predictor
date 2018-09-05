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
from sklearn.metrics import confusion_matrix
from sklearn import metrics

from utility import save_load_Keras, plot_tSNE, keras_event_callBack, io
import train_test_config as conf
from feature_extraction import label_generator as label_gen
from feature_extraction import label_generator_rahul as label_gen_r
from feature_extraction import feature_engineering
from feature_extraction import feature_extraction
from model.nn_model_rahul_keras import nn_model
from model import ensemble_model
import generate_rahual_features



if __name__ == '__main__':
    
    choose_TopN_features = 10
    train_set, test_set, test_raw_office = generate_rahual_features.generate_rahual_featrues(choose_TopN_features)
    
    #Regression
    io.c_print('Start regressor training')
    m = ensemble_model.models(train_set[0], train_set[2], test_set, 5)
    model = m.model_xgb
    model.fit(np.log(train_set[0]), np.log(train_set[2]))  
    io.c_print('Finish regressor training') 
    
    #Evaluation
    pred_test = model.predict(test_set[0])
    #io.c_print('RMSE:{}'.format(m.rmse(test_set[2], pred_test)))
    io.c_print('RMSE:{}'.format(m.rmse(test_set[2], np.expm1(pred_test))))
    
    #Evaluation classification error
    label_test_cat = label_gen.transform_delay2category(pd.DataFrame(test_set[2]))
    pred_test_cat = label_gen.transform_delay2category(pd.DataFrame(np.expm1(pred_test)))    
    io.c_print('Accuracy reg_clf:{}'.format(metrics.accuracy_score(label_test_cat, pred_test_cat)))
    c_matrix_rclf = confusion_matrix(label_test_cat, pred_test_cat)
    io.c_print(c_matrix_rclf)



    
    
    