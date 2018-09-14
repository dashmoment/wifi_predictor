import sys
sys.path.append("../")

import pandas as pd
import numpy as np
import pickle
from feature_extraction import feature_engineering
from feature_extraction import label_generator_rahul as label_gen_r


def split_dataset(output="attenuator"):

    with open('../data/raw_data_sample_ma_w_time_sub.pkl', 'rb') as input:

        train = pickle.load(input)
        label_train = pickle.load(input)
        test = pickle.load(input)
        label_test = pickle.load(input)
        train_off = pickle.load(input)
        label_train_off = pickle.load(input)
        test_off = pickle.load(input)
        label_test_off = pickle.load(input)

    if output == "attenuator":

        # Use only attenuator data to split into training and testing set
        train, label_train = label_gen_r.sort_by_time(train, label_train)
        test, label_test = label_gen_r.sort_by_time(test, label_test)
    
    if output == "office":
        
        # Use only office data to split into training and testing set
        train, label_train = label_gen_r.sort_by_time(train_off, label_train_off)
        test, label_test = label_gen_r.sort_by_time(test_off, label_test_off)
    
    if output == "combine":

        # combine attenuator and office data
        train = pd.concat([train, train_off])
        label_train = pd.concat([label_train['delay_mean'], label_train_off['delay_mean']])
        label_train = pd.DataFrame(label_train).rename(columns={'Delay-mean': 'delay_mean'})
        label_train = {'delay_mean': label_train}
        
        test = pd.concat([test, test_off])
        label_test = pd.concat([label_test['delay_mean'], label_test_off['delay_mean']])
        label_test = pd.DataFrame(label_test).rename(columns={'Delay-mean': 'delay_mean'})
        label_test = {'delay_mean': label_test}

        train, label_train = label_gen_r.sort_by_time(train, label_train)
        test, label_test = label_gen_r.sort_by_time(test, label_test)

    # merge all office data as a pure testing set
    test_off = pd.concat([train_off, test_off])
    label_test_off = pd.concat([label_train_off['delay_mean'], label_test_off['delay_mean']])
    label_test_off = pd.DataFrame(label_test_off).rename(columns={'Delay-mean': 'delay_mean'})
    label_test_off = {'delay_mean': label_test_off}
    
    test_off, label_test_off = label_gen_r.sort_by_time(test_off, label_test_off)

    return train, label_train, test, label_test, test_off, label_test_off


