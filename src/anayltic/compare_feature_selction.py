import sys
sys.path.append("../")
import os
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import model_from_json

from feature_extraction import feature_engineering
from sklearn.feature_selection import SelectKBest, f_regression
from feature_extraction import label_generator_rahul as label_gen_r


with open('../../data/raw_data_wo_time_sub.pkl', 'rb') as input:

        train_leagacy = pickle.load(input)
        label_train = pickle.load(input)
        test_raw = pickle.load(input)
        label_test_raw = pickle.load(input)
        
train_leagacy = feature_engineering.binding(train_leagacy)
label_train_dm =  label_gen_r.TransferToOneHotClass(label_train['delay_mean'])
anova_filter = SelectKBest(f_regression, k=15).fit(train_leagacy, label_train['delay_mean'])
mask = anova_filter.get_support(indices=True)
print('Selected features from regression: {}'.format(train_leagacy.columns[mask]))

#Concat data and label
data = pd.concat([train_leagacy,label_train['delay_mean']], axis=1)
#Calculate corr heatmap and plot
corr = data.corr().abs()
plt.subplots(figsize=(30,20))
hmap = sns.heatmap(
                    corr, 
                    xticklabels=corr.columns,
                    yticklabels=corr.columns
                    )

figure = hmap.get_figure()
figure.savefig("corr_heatmap.png", dpi=600)

sorted_corr = corr.sort_values('Delay-mean', ascending=False)
print('Selected features from corr: {}'.format(sorted_corr.index[1:11]))

#Train with rahual model to compare fetature selection
import keras
from utility import keras_event_callBack, calculate_classWeight
from model.nn_model_rahul_keras import nn_model

train = train_leagacy[sorted_corr.index[1:11]]
train = train.sample(frac=1.0)    
label_train_shuffle = label_train['delay_mean'].iloc[train.index]
train_set = [
             train,
             label_gen_r.TransferToOneHotClass(label_train_shuffle)
            ]
 # normalization
train_set[0] = (train_set[0] - train_set[0].mean())/train_set[0].std()

#Build Network

savePath_root =  '../../trained_model/nn_model_corr_feature_selection/'
if os.path.exists(savePath_root):
    
    print(" Load Trained Model")
    json_file = open(os.path.join(savePath_root, 'graph.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    rahul_model = model_from_json(loaded_model_json)
    rahul_model.load_weights(os.path.join(savePath_root, 'weight.h5'))
    
    rahul_model.summary()
    
else:
    print("Model not exist. Build new model")
    os.mkdir(savePath_root)
    rahul_model = nn_model(train_set[0].shape[1:])
    rahul_model.summary()
adam = keras.optimizers.Adam(lr=0.0005, epsilon=1e-8)
rahul_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])


   

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
                   class_weight=calculate_classWeight.calculate_classWeight(train_set[1]),
                   callbacks = [saveModel_cb, tensorBoard_cb] 
                   )


