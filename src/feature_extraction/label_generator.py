import pandas as pd
import numpy as np

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

def transform_delay2category(delay):
    
    delay_cat = pd.cut(
                        delay.stack(),
                        [-np.inf,5,10,20, np.inf],
                        labels = [0,1,2,3]
                        ).reset_index(drop=True)
    
    return delay_cat