import sys
sys.path.append("../")
import os
import numpy as np
import pandas as pd
import pickle
from feature_extraction import feature_engineering
from sklearn.feature_selection import SelectKBest, f_regression
from feature_extraction import label_generator_rahul as label_gen_r

def generate_rahual_featrues(topN_feature=10):
    
    with open('../../data/raw_data_wo_time_sub.pkl', 'rb') as input:

        train = pickle.load(input)
        label_train = pickle.load(input)
        test_raw = pickle.load(input) #Office Data
        label_test_raw = pickle.load(input)
        
    total = pd.concat([train, test_raw])
    label_total = pd.concat([label_train['delay_mean'], label_test_raw['delay_mean']])
    label_total = pd.DataFrame(label_total).rename(columns={'Delay-mean': 'delay_mean'})
        
    train = feature_engineering.binding(train)
    train, label_train, test, label_test = label_gen_r.RandomSample(total, label_total, fraction=0.8)
    
    if topN_feature == -1: topN_feature = len(train.columns) #Get all features
    print(topN_feature)
    anova_filter = SelectKBest(f_regression, k=topN_feature).fit(train, label_train['delay_mean'])
    mask = anova_filter.get_support(indices=True)
    print('Selected features: {}'.format(train.columns[mask]))
      
    
    train_set = [train.iloc[:, mask],
                 label_gen_r.TransferToOneHotClass(label_train['delay_mean']),
                 label_train['delay_mean']]
    test_set = [test.iloc[:, mask],
                label_gen_r.TransferToOneHotClass(label_test['delay_mean']),
                label_test['delay_mean']]
    test_raw_set = [test_raw.iloc[:, mask],
                    label_gen_r.TransferToOneHotClass(label_test_raw['delay_mean']),
                    label_test_raw['delay_mean']]
    
    
    # normalization
    train_set[0] = (train_set[0] - train_set[0].mean())/train_set[0].std()
    test_set[0] = (test_set[0] - test_set[0].mean())/test_set[0].std()
    test_raw_set[0] = (test_raw_set[0] - test_raw_set[0].mean())/test_raw_set[0].std()
    
    return train_set, test_set, test_raw_set
    

