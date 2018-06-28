#Train and test from different set
#Add singal value
import sys
sys.path.append("../")
import matplotlib.pyplot as plt
#plt.switch_backend('agg')
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from scipy import stats
from scipy.stats import norm, skew
from xgboost import plot_importance
from sklearn.metrics import confusion_matrix
import xgboost as xgb

from utility import io
import train_test_config as conf
from feature_extraction import feature_extraction
from feature_extraction import label_generator as label_gen
from model import ensemble_model 
#Manual Split training and test set


config = conf.train_test_config('Read_Collection_train_ct', 'Read_Collection_test_ct')
fext = feature_extraction.feature_extraction()

#fdata = fext._getData(config.train) 

if __name__ == '__main__': 
        
    #Generator train & test data by configuration 
    train, label_train = fext.generator(config.train, time_step=15)
    test, label_test = fext.generator(config.test, time_step=15)
    
    #Divide from training set
    '''
    portion = int(len(train)*0.8)
    test = train[portion:]
    label_test = label_train[portion:]
    train = train[:portion]
    label_train = label_train[:portion]
    '''
    
    io.c_print('Finish loading data')
    
    label_train = label_train['delay_mean_log']
    label_test = label_test['delay_mean_log']
 
    
    #Regression
    io.c_print('Start regressor training')
    m = ensemble_model.models(train, label_train, test, 5)
    model = m.model_lgb
    model.fit(train, label_train)    
    io.c_print('Finish regressor training')    
    
    #Evaluation
    pred_test = model.predict(test)
    io.c_print('RMSE:{}'.format(m.rmse(np.expm1(label_test), np.expm1(pred_test))))
    
    #Evaluation classification error
    label_test_cat = pd.DataFrame(np.expm1(label_test))
    pred_test_cat = pd.DataFrame(np.expm1(pred_test))
 
    label_test_cat = label_gen.transform_delay2category(label_test_cat)
    pred_test_cat = label_gen.transform_delay2category(pred_test_cat)
       
    io.c_print('Recall reg_clf:{}'.format(metrics.recall_score(label_test_cat, pred_test_cat, average = 'micro')))
    c_matrix_rclf = confusion_matrix(label_test_cat, pred_test_cat)
    io.c_print(c_matrix_rclf)
    
#    io.c_save_image(plt.plot, y_valid, filename='tvalid.png')
#    io.c_save_image(plt.plot, y_pred, filename='tpred.png')
#    io.c_save_image(plt.scatter, y_valid, y_pred, filename='tvalid_pred.png')
#    
#    importance=pd.Series(test_model.feature_importances_,index=train.columns).sort_values(ascending=False)
#    importance.to_csv('importance.csv')

    
    ''' 
    Test classfier
    '''
    io.c_print('Start classfier training')
    model = xgb.XGBClassifier(max_depth=3, learning_rate=0.05 ,n_estimators=2000, silent=True)    
    label_train_cat = label_gen.transform_delay2category(pd.DataFrame(np.expm1(label_train)))
    model.fit(train, label_train_cat)
    
    pred_test_clf = model.predict(test)    
    io.c_print('Recall clf: {}'.format(metrics.recall_score(label_test_cat, pred_test_clf, average = 'micro')))    
    xg_train_c_matrix = confusion_matrix(label_test_cat, pred_test_clf)
    io.c_print(xg_train_c_matrix)  
    io.c_print('Finish classfier training') 

