import mongodb_api as db
import pandas as pd
import sys
import numpy as np
import model_zoo as mz


from matplotlib import pyplot as plt

h5data = pd.HDFStore('../data/scan_data_1061129.h5')
raw_data= h5data["raw_data"]

data = []
label = []

old_label = 0

data_type = "Sigval_Std"
label_type = "Spectralscan_mean"

#data_type = "survey_data"
#label_type = "survey_mean"

for i in range(len(raw_data)):
    
    new_label = raw_data[label_type][i]
    
    if abs(new_label - old_label) > 0 and new_label < 100:
        data.append(raw_data[data_type][i])
        label.append(raw_data[label_type][i])
    
    old_label = raw_data[label_type][i]
#data = list(raw_data["Spectralscan_data"])
#label =  raw_data["Spectralscan_mean"]

train_set = [data, label]
#results = mz.ridge(train_set, train_set, alpha = 0.001)
#results = mz.svr_model(train_set, train_set, epsilon=0.01)
results, nn_loss = mz.Spectralscan_NN(train_set, train_set)


plt.figure(figsize=(15, 10))
x_axis = range(len(data))
label_p = plt.plot(x_axis,results)
plt.show()


mse = np.sqrt(np.mean(np.power(results -label,2)))

print("MSE Score: ", mse)


plt.figure(figsize=(15, 10))
x_axis = range(len(data))
l = plt.scatter(x_axis, label, facecolors='none', marker='o',edgecolors='r', label='label')
pre = plt.scatter(x_axis, results, facecolors='none',marker='o',edgecolors='g', label='prediction')

plt.legend((pre,l), ("Prediction", "label"), fontsize=15)
plt.show()

pre = plt.scatter(x_axis, results, facecolors='none',marker='o',edgecolors='g', label='prediction')

plt.figure(figsize=(15, 10))
plt.scatter(x_axis, np.sqrt(np.power(results -label,2)),facecolors='none',marker='o',edgecolors='g')
plt.show()



###Feature and label relation plot

busy = []
rcv = []
tx =[]

plt.figure(figsize=(15, 10))
x_axis = range(len(data))
label_p = plt.plot(x_axis,label)
plt.show()

for j in range(len(data[0])):
    
    plot_data = []
    for i in range(len(data)):
        
        plot_data.append(data[i][3])
#        rcv.append(data[i][4])
#        tx.append(data[i][5])
    plt.figure(figsize=(15, 10))
    busy_p = plt.plot(x_axis,plot_data)
    plt.show()

#pair = sorted(zip(label, busy, rcv, tx))

#label, busy, rcv, tx = zip(*pair)


#plt.figure(figsize=(15, 10))
#x_axis = range(len(data))
#label_p = plt.plot(x_axis,label)
#busy_p = plt.plot(x_axis,busy)
#rcv_p = plt.plot(x_axis, rcv)
##tx_p = plt.plot(x_axis, tx)
#plt.legend((busy_p,rcv_p, label_p), ("Busy", "RCV", "labels"), fontsize=15)
#plt.show()


