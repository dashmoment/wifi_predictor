import sys
sys.path.append("../")
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
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
import statsmodels.api as sm
import statsmodels.formula.api as smf

from tensorflow.python.framework import ops
from scipy import stats
from scipy.stats import norm, skew

import xgboost as xgb
from xgboost import plot_importance

from utility import io
import train_test_config as conf
from feature_extraction import feature_extraction
from feature_extraction import feature_extraction_rahul
from feature_extraction import label_generator as label_gen
from model import ensemble_model
from model import nn_model_rahul

import pickle
import importlib
from datetime import datetime
# importlib.reload(nn_model_rahul)
# importlib.reload(feature_extraction_rahul)

# load data
with open('../data/raw_data.pkl', 'rb') as input:
    
    train = pickle.load(input)
    label_train = pickle.load(input)
    test = pickle.load(input)
    label_test = pickle.load(input)


if __name__ == '__main__':

    total = pd.concat([train, test])
    label_total = pd.concat([label_train['delay_mean'], label_test['delay_mean']])
    label_total = pd.DataFrame(label_total).rename(columns={'Delay-mean': 'delay_mean'})

    train = total.sample(frac = 0.8)
    sample_idx = train.index
    label_train = label_total.iloc[sample_idx]
    
    test = total[~total.index.isin(sample_idx)]
    label_test = label_total[~total.index.isin(sample_idx)]
    
    
    # feature selection
    anova_filter = SelectKBest(f_regression, k = 4).fit(
        train, label_train['delay_mean'])
    mask = anova_filter.get_support(indices=True)
 
    train_set = [np.array(train.iloc[:, mask]),
                 np.array(label_train['delay_mean'])]
    test_set = [np.array(test.iloc[:, mask]),
                np.array(label_test['delay_mean'])]
    
    # apply NN model on selected data
    ops.reset_default_graph()
    prediction, loss = nn_model_rahul.general_NN(
        train_set, test_set, feature=4, batch_size=32, Max_epoch=50)
    
    # Evaluation classification error
    label_test_cat = label_gen.transform_delay2category(
        pd.DataFrame(label_test['delay_mean']))
    pred_test_cat = label_gen.transform_delay2category(pd.DataFrame(prediction))

    print('Recall reg_clf:{}'.format(metrics.recall_score(
        label_test_cat, pred_test_cat, average='micro')))
    print(metrics.accuracy_score(label_test_cat, pred_test_cat))





    
