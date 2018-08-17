import os
from os import path
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

from tensorflow.python.framework import ops
from scipy import stats
from scipy.stats import norm, skew
import xgboost as xgb
from xgboost import plot_importance
import keras
import keras.backend as K

from feature_extraction import label_generator as label_gen
from feature_extraction import label_generator_rahul as label_gen_r
from feature_extraction import feature_engineering
from model.nn_model_rahul_keras import nn_model
from utility import io
from utility import save_load_Keras
from utility import keras_event_callBack

import pickle
import importlib
from datetime import datetime

# importlib.reload(nn_model_rahul)
# importlib.reload(feature_extraction_rahul)

ROOT_DIR = path.dirname(path.abspath(__file__))

with open('../data/raw_data_wo_time_sub.pkl', 'rb') as input:

    train = pickle.load(input)
    label_train = pickle.load(input)
    test_raw = pickle.load(input)
    label_test_raw = pickle.load(input)


train = feature_engineering.binding(train)
test_raw = feature_engineering.binding(test_raw)


# Use AP only
ap_col = [col for col in train.columns if 'AP' in col]
train = train[ap_col]
test_raw = test_raw[ap_col]

# Drop error columns
err_col = [col for col in train.columns if 'ERRORS' in col or 'ERR' in col or 'Error' in col]
train = train.drop(err_col, axis=1)
test_raw = test_raw.drop(err_col, axis=1)

# Use only attenuator data to split into training and testing set
#train, label_train, test, label_test = label_gen_r.random_sample(train, label_train, fraction=0.8)


# Combine attenuator and office data to split into training and testing set
total = pd.concat([train, test_raw])
label_total = pd.concat([label_train['delay_mean'], label_test_raw['delay_mean']])
label_total = pd.DataFrame(label_total).rename(columns={'Delay-mean': 'delay_mean'})

train, label_train, test, label_test = label_gen_r.random_sample(total, label_total, fraction=0.8)

print(train.isnull().any())
print(train.isnull().sum())


def neural_network(num_feature=8, lr=0.0005, batch_size=64, epochs=1):

    # feature selection
    if num_feature > train.shape[1]:
        num_feature = train.shape[1]
    anova_filter = SelectKBest(f_regression, k=num_feature).fit(train, label_train['delay_mean'])
    mask = anova_filter.get_support(indices=True)
    print('Selected features: {}'.format(train.columns[mask]))

    train_set = [train.iloc[:, mask],
                 label_gen_r.transfer_to_one_hot_class(label_train['delay_mean'])]
    test_set = [test.iloc[:, mask],
                label_gen_r.transfer_to_one_hot_class(label_test['delay_mean'])]
    test_raw_set = [test_raw.iloc[:, mask],
                    label_gen_r.transfer_to_one_hot_class(label_test_raw['delay_mean'])]

    # normalization
    train_set[0] = (train_set[0] - train_set[0].mean())/train_set[0].std()
    test_set[0] = (test_set[0] - test_set[0].mean())/test_set[0].std()
    test_raw_set[0] = (test_raw_set[0] - test_raw_set[0].mean())/test_raw_set[0].std()

    # apply NN model on selected data
    ops.reset_default_graph()

    wifi = nn_model(train_set[0].shape[1:])

    adam = keras.optimizers.Adam(lr=lr, epsilon=1e-8)
    adam_decay = keras.optimizers.Adam(lr=lr, decay=1e-6, epsilon=1e-8)
    sgd = keras.optimizers.SGD(lr=lr, momentum=0.9)
    sgd_decay = keras.optimizers.SGD(lr=lr, momentum=0.9, decay=1e-6)
    wifi.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    # weighted loss
    y_train_classes = label_gen_r.transform_delay2category(label_train['delay_mean'])
    y_train_classes = pd.DataFrame(y_train_classes, columns=['class'])
    n = len(y_train_classes)
    y_train_classes_count = n / y_train_classes['class'].value_counts().sort_index(axis=0)

    class_weight = {}
    for idx, value in enumerate(y_train_classes_count):
        class_weight[idx] = value

    # callback function to save the trained model
    graph_path = path.join(ROOT_DIR, 'saved_models', 'focal_loss', 'model.json')
    weight_path = path.join(ROOT_DIR, 'saved_models', 'focal_loss', 'model.h5')
    filepaths = [graph_path, weight_path]
    for filepath in filepaths:
        dirname = os.path.dirname(filepath)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

    saveModel = keras_event_callBack.saveModel_Callback(save_epoch=50,
                                                        model=wifi,
                                                        graph_path=graph_path,
                                                        weight_path=weight_path)

    # fit the NN model
    history = wifi.fit(train_set[0], train_set[1],
                       epochs=epochs,
                       validation_split=0.2,
                       batch_size=batch_size,
                       class_weight=class_weight,
                       callbacks=[saveModel])
    score = wifi.evaluate(test_set[0], test_set[1], verbose=0)
    print('Test loss: {} - Test accuracy: {}'.format(score[0], score[1]))
    score = wifi.evaluate(test_raw_set[0], test_raw_set[1], verbose=0)
    print('Test_raw loss: {} - Test_raw accuracy: {}'.format(score[0], score[1]))
    print('Selected features: {}'.format(train.columns[mask]))

    # record training history
    plt.switch_backend('agg')
    io.show_train_history(history)

    # confusion matrix on test and test_raw
    y_predict_prob = wifi.predict(test_set[0])
    y_predict_classes = y_predict_prob.argmax(axis=-1)
    y_test_classes = label_gen_r.transform_delay2category(label_test['delay_mean'])
    print('Confusion matrix for split testing data')
    print(pd.crosstab(y_test_classes, y_predict_classes,
                      rownames=['label'], colnames=['prediction']), '\n')

    y_predict_prob = wifi.predict(test_raw_set[0])
    y_predict_classes = y_predict_prob.argmax(axis=-1)
    y_test_classes = label_gen_r.transform_delay2category(label_test_raw['delay_mean'])
    print('Confusion matrix for testing data')
    print(pd.crosstab(y_test_classes, y_predict_classes,
                      rownames=['label'], colnames=['prediction']))


def regression():
    # not complete yet
    pass


# define classification accuracy for regression
def soft_acc(y_true, y_pred):

    return K.mean(K.equal(K.round(y_true), K.round(y_pred)))


if __name__ == '__main__':
    neural_network(num_feature=10, lr=0.001, batch_size=64, epochs=300)
