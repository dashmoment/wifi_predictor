from mongodb_api import mongodb_api
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew
from scipy.special import boxcox1p
from sklearn import metrics

def plot_dist_QQ_skew(display, data):
    
    plt.figure()
    sns.distplot(data)
    plt.show()    
    plt.figure()
    stats.probplot(data, plot=plt)
    plt.show()
    
    print('{} skewness: {}'.format(display, skew(data)))
    

def delay_analysis(MLdf):
    
       
    '''
    Watch delay distribution, and fix skewness
    '''
    
    #Check skewness
    
    MLdf['Delay-mean_log'] = np.log1p(MLdf['Delay-mean'])
    #plot_dist_QQ_skew('MLdfDelay-mean', MLdf['Delay-mean'])
    #plot_dist_QQ_skew('MLdfDelay-mean log', MLdf['Delay-mean_log'])
    #box-cox doesn't help
    
    '''
    lam = 0.15
    MLdf['Delay-mean_boxcox'] = boxcox1p(MLdf['Delay-mean'], lam)
    stats.probplot(MLdf['Delay-mean_boxcox'], plot=plt)    
    plot_dist_QQ_skew('MLdfDelay-mean BoxCox', MLdf['Delay-mean_boxcox'])
    '''
    
    #Delay max
    MLdf['Delay-max_log'] = np.log1p(MLdf['Delay-max'])
    #plot_dist_QQ_skew('Delay-max', MLdf['Delay-max'])
    #plot_dist_QQ_skew('Delay-max log', MLdf['Delay-max_log'])
    
    '''
    lam = 0.15
    MLdf['Delay-max_boxcox'] = boxcox1p(MLdf['Delay-max'], lam)
    stats.probplot(MLdf['Delay-max_boxcox'], plot=plt)    
    plot_dist_QQ_skew('MLdfDelay-max BoxCox', MLdf['Delay-max_boxcox'])
    '''
   
    
    
    
    
    
    
    
    