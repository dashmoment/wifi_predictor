import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import os
import random
#import model_zoo as mz

#import tensorflow.contrib.eager as tfe
from sklearn.metrics import confusion_matrix
from scipy.fftpack import fft
from matplotlib import pyplot as plt
import copy
import mongodb_api as db
import model_zoo as mz

# Training data
#['sigval', 'busy_time', 'Rssi', 'Portion']
#['sigval_std','sigval', 'busy_time', 'Rssi', 'Portion']

feature_size = 56
def preprocess_data(raw_data, feature_type = ['sigval_std'], raw_data_type = ['STA']):
    
    data = []
    
    for i in range(len(feature_type)):
        
        #tmp_data = raw_data[raw_data_type[0]][feature_type[i]] + raw_data[raw_data_type[1]][feature_type[i]] 
        tmp_data = raw_data[raw_data_type[0]][feature_type[i]]
        tmp_data = np.around(np.array(tmp_data).astype(np.float32), decimals=3)
        tmp_data = tmp_data.reshape(-1,np.shape(tmp_data)[-1])
        
        if  feature_type[i] != 'Portion':
            norm = np.mean(tmp_data, axis=0)
            std = np.std(tmp_data, axis=0)
            tmp_data = (tmp_data - norm)/std
        
        #print(norm, std)
        
        #print(np.shape(tmp_data))
        
        if feature_type[i] == 'sigval_std' or feature_type[i] == 'sigval':
    
            tmp_data = fft(tmp_data)
            data.append(tmp_data)
            #data.append(tmp_data[:,1:56].reshape(-1,55))
            
            
        else:
            data.append(tmp_data)
            
       
        #print(np.shape(data))
    
        
    return np.concatenate(data, axis=-1)
        

data_feature = 'Portion'
label_feature = 'Ping_mean'

h5data = pd.HDFStore('../data/ProcessData1070108/training_data_t1.h5')
raw_data= h5data["raw_data"]

slice_data = preprocess_data(raw_data).reshape(-1,feature_size)[0:600]
#slice_label = raw_data['STA'][label_feature] + raw_data['AP'][label_feature]
slice_label = raw_data['STA'][label_feature][0:600]
slice_label = slice_label
#raw_data = None
print("Load training done")

h5data = pd.HDFStore('../data/ProcessData1070108/testing_data_t1.h5')
raw_data= h5data["raw_data"]
testslice_data = preprocess_data(raw_data).reshape(-1,feature_size)[0:600]
#testslice_label = raw_data['STA'][label_feature] + raw_data['AP'][label_feature]
testslice_label = raw_data['AP'][label_feature][0:600]
testslice_label = testslice_label
#raw_data = None    
print("Load testing done")


####generate One hot ecoded label

def one_hot_label_generator(labels, divisions = [0,2,5]):

    def check_in_range(label, divisions):

        if label>=divisions[0] and label < divisions[1]:
            return True
        else:
            return False

    
    oh_label = []
    for l_idx in range(len(labels)):
        for idx in range(len(divisions)-1):
            
            if check_in_range(labels[l_idx],[divisions[idx], divisions[idx+1]]): 
                oh_label.append(idx)
                break
        if len(oh_label) != l_idx + 1:
            oh_label.append(len(divisions)-1)

       
    return oh_label




###SVR test
from sklearn.svm import SVR
clf = SVR(C=1.0, epsilon=0.3)
clf.fit(slice_data, slice_label) 

pre_train = clf.predict(slice_data)
pre_test = clf.predict(testslice_data)

#plt.plot(pre_train)
#plt.show()
#plt.plot(slice_label)
#plt.show()
#plt.plot(pre_test)
#plt.show()
#plt.plot(testslice_label)
#plt.show()


###Gradient

g_train_data = slice_data
g_test_data = testslice_data
train_label = slice_label
test_label = testslice_label

###Moving average
"""
period = 1
train_data = []
train_label = []
test_data = []
test_label = []
for i in range(period,len(g_train_data)):
        train_data.append(np.mean(g_train_data[i-period:i]))
        train_label.append(slice_label[i])
        test_data.append(np.mean(g_test_data[i-period:i]))
        test_label.append(testslice_label[i])
        

g_train_data = []
g_test_data = []

for i in range(1,len(train_data)):
        g_train_data.append(train_data[i] - train_data[i-1])
        g_test_data.append(test_data[i] - test_data[i-1])

g_train_data = train_data
g_test_data = test_data

"""



plt.plot(g_train_data)
plt.show()
plt.plot(train_label)
plt.show()
oh_train_label = one_hot_label_generator(train_label, divisions = [0,3,5])
plt.plot(oh_train_label)
plt.show()
plt.plot(g_test_data)
plt.show()
plt.plot(test_label)
plt.show()
oh_test_label = one_hot_label_generator(test_label, divisions = [0,3,5])
plt.plot(oh_test_label)
plt.show()

###SVC test
from sklearn.svm import SVC
feature_size = 1

train_data = np.array(g_train_data).reshape(-1,feature_size)
test_data = np.array(g_test_data).reshape(-1,feature_size)
oh_train_label = oh_train_label[:]
oh_test_label = oh_test_label[:]
train_data = slice_data
test_data = testslice_data

clf = SVC()
clf.fit(train_data, oh_train_label)

prediction_train = clf.predict(train_data)
accuracy_train = np.mean(np.equal(oh_train_label, prediction_train).astype(np.float32))
train_c_matrix = confusion_matrix(oh_train_label, prediction_train)
plt.imshow(train_c_matrix)
plt.show()

prediction_test = clf.predict(test_data)
accuracy_test = np.mean(np.equal(oh_test_label, prediction_test).astype(np.float32))
test_c_matrix = confusion_matrix(oh_test_label, prediction_test)
plt.imshow(test_c_matrix)
plt.show()


##### XGBoost trial#############
from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(train_data, oh_train_label)
print(model)
xg_prediction_train = model.predict(train_data)
xg_accuracy_train = np.mean(np.equal(oh_train_label, xg_prediction_train).astype(np.float32))
xg_train_c_matrix = confusion_matrix(oh_train_label, xg_prediction_train)
plt.imshow(xg_train_c_matrix)
plt.show()

xg_prediction_test = model.predict(test_data)
xg_accuracy_test = np.mean(np.equal(oh_test_label, xg_prediction_test).astype(np.float32))
xg_test_c_matrix = confusion_matrix(oh_test_label, xg_prediction_test)
plt.imshow(xg_test_c_matrix)
plt.show()































