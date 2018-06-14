
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import os
import random
import model_zoo as mz

#import tensorflow.contrib.eager as tfe
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import copy

def one_hot(y_):
    # Function to encode output labels from number indexes 
    # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    
    y_ = y_.reshape(len(y_))
    n_values = np.max(y_) + 1
    
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]  # Returns FLOATS


### Data Prerpocess

# Training data
#['sigval', 'busy_time', 'Rssi']
feature_size = 1
def preprocess_data(raw_data, feature_type = ['sigval'], raw_data_type = ['STA', 'AP']):
    
    data = []
    
    for i in range(len(feature_type)):
        
        tmp_data = raw_data[raw_data_type[0]][feature_type[i]] + raw_data[raw_data_type[1]][feature_type[i]] 
        tmp_data = np.around(np.array(tmp_data).astype(np.float32), decimals=3)
         
        #flat_data =tmp_data.reshape(-1,1)
        #norm = np.mean(flat_data, axis=0)
        #std = np.std(flat_data, axis=0)
        #tmp_data = (tmp_data - norm)/std
        
        #print(norm, std)
        
        if len(tmp_data.shape) != 2: tmp_data = tmp_data.reshape(-1,56)
        print( len(tmp_data.shape))
        
        data.append(tmp_data)
        
    return np.concatenate(data, axis=-1)
        

data_feature = 'sigval'
label_feature = 'FER'

h5data = pd.HDFStore('../data/ProcessData1228/training_data_t1.h5')
raw_data= h5data["raw_data"]

slice_data = preprocess_data(raw_data).reshape(-1,feature_size)
slice_label = raw_data['STA'][label_feature] + raw_data['AP'][label_feature]
slice_label = slice_label
#raw_data = None
print("Load training done")

h5data = pd.HDFStore('../data/ProcessData1228/testing_data_t1.h5')
raw_data= h5data["raw_data"]
testslice_data = preprocess_data(raw_data).reshape(-1,feature_size)
testslice_label = raw_data['STA'][label_feature] + raw_data['AP'][label_feature]
testslice_label = testslice_label
#raw_data = None    
print("Load testing done")

new_label = []
for i in range(len(slice_label)): 
    
    tmp = int(slice_label[i]//0.2)
    if tmp < 2: new_label.append(tmp) 
    else: new_label.append(2)  
#slice_label  = copy.deepcopy(one_hot(np.array(new_label))) 
slice_label  = copy.deepcopy(np.array(new_label)) 

new_label = []
for i in range(len(testslice_label)): 
    tmp = int(testslice_label[i]//0.2)
    if tmp < 2: new_label.append(tmp) 
    else: new_label.append(2)
#testslice_label  = copy.deepcopy(one_hot(np.array(new_label)))
testslice_label  = copy.deepcopy(np.array(new_label))

from sklearn.svm import SVC
clf = SVC()
clf.fit(slice_data, slice_label) 
prediction_train = clf.predict(slice_data)
accuracy_train = np.mean(np.equal(slice_label, prediction_train).astype(np.float32))
prediction_test = clf.predict(testslice_data)
accuracy_test = np.mean(np.equal(testslice_label, prediction_test).astype(np.float32))



"""

############### Build Model###################

#feature_size = len(slice_data[0][0])
#time_step = len(slice_data[0])

num_hidden = 32
num_classes = 3

weights = {
    #'hidden': tf.get_variable("hidden_w",[feature_size, num_hidden],initializer=tf.random_normal_initializer(1.0, 0.02)),
    #'out':  tf.get_variable("out_w",[num_hidden, num_classes],initializer=tf.random_normal_initializer(1.0, 0.02)) 
       
        
    }
biases = {
    #'hidden': tf.get_variable("hidden_b",[num_hidden],initializer=tf.random_normal_initializer(1.0, 0.02)),
    #'out':  tf.get_variable("out_b",[num_classes],initializer=tf.random_normal_initializer(1.0, 0.02)) 
    }


with tf.variable_scope("lstm", reuse=None):
    
    
    
    #inputs = tf.placeholder(tf.float32, [None, time_step, feature_size])
    inputs = tf.placeholder(tf.float32, [None, feature_size])
    labels = tf.placeholder(tf.float32, [None, num_classes])
    lr = tf.placeholder(tf.float32, name='learning_rate')
    dropout = tf.placeholder(tf.float32, name='dropout')

   
    #FC net test
    net = mz.fc_layer(inputs, num_hidden, "fc1")
    net = mz.fc_layer(net, num_hidden, "fc2")
    net = mz.fc_layer(net, num_hidden, "fc3")
    #net = tf.nn.dropout(net, dropout)
    logits = mz.fc_layer(net, 3, "logits", activat_fn=None)
    
     

    #ustack_in = tf.unstack(inputs, time_step, 1)
    
    ### 2-layer lstm
    #lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(num_hidden, forget_bias=1.0, state_is_tuple=True)
    #lstm_cell_1 = rnn.DropoutWrapper(lstm_cell_1,  output_keep_prob=dropout)
    #lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(num_hidden, forget_bias=1.0, state_is_tuple=True)
    #lstm_cell_2 = rnn.DropoutWrapper(lstm_cell_2,  output_keep_prob=dropout)
    #lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
    
    #outputs, states = tf.contrib.rnn.static_rnn(lstm_cells, ustack_in, dtype=tf.float32)
    #logits = tf.matmul(outputs[-1], weights['out']) + biases['out']
    
    
    #lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    #lstm_cell = rnn.DropoutWrapper(lstm_cell,  output_keep_prob=dropout)
    #outputs, states = rnn.static_rnn(lstm_cell, ustack_in, dtype=tf.float32)
    #logits = tf.matmul(outputs[-1], weights['out']) + biases['out']
    



### aux function
#raw_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
#regularization
tv = tf.trainable_variables()
regularization_cost = 0.001*tf.reduce_sum([ tf.nn.l2_loss(v) for v in tv])
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
loss = cross_entropy + regularization_cost
optimizer = tf.train.RMSPropOptimizer(learning_rate=lr)
w_grad = optimizer.compute_gradients(loss)
train_op = optimizer.apply_gradients(w_grad)

tlabel = tf.argmax(tf.nn.softmax(logits), axis=1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(tf.nn.softmax(logits), axis=1), tf.argmax(labels,1)), tf.float32))




with tf.name_scope('train_summary'):
    
    tf.summary.scalar("loss", loss, collections=['train'])
    tf.summary.scalar("cross_entropy", cross_entropy, collections=['train'])
    tf.summary.scalar("regularization", regularization_cost, collections=['train'])
    tf.summary.scalar("accuracy", accuracy, collections=['train'])
     
    for grad, var in w_grad: 
        tf.summary.histogram(var.name + '/gradient_l1', grad, collections=['train']) 
        
    train_variables = tf.trainable_variables()
    generator_variables = [v for v in train_variables]
    for v in generator_variables:
        tf.summary.histogram(v.name , v, collections=['train']) 
    merged_summary_train = tf.summary.merge_all('train') 
    
with tf.name_scope('test_summary'):
    
    tf.summary.scalar("loss", loss, collections=['test'])
    tf.summary.scalar("cross_entropy", cross_entropy, collections=['test'])
    tf.summary.scalar("regularization", regularization_cost, collections=['test'])
    tf.summary.scalar("accuracy", accuracy, collections=['test'])
    merged_summary_test = tf.summary.merge_all('test')  
    
    
# Session run
    
    
def create_sum_and_ckpt(sess, checkpoint_dir, log_dir, ckpt_name):
    
    saver = tf.train.Saver()
   
    print(" [*] Reading checkpoints...")

    
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    
    
    if ckpt  and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        
        print(" [*] Load SUCCESS")
        
    else:
        print(" [!] Load failed...")   
        
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

    
    return saver , summary_writer  
    


def shuffle_data(data, label):
    
    data_pair = list(zip(data,label))
    random.shuffle(data_pair)
    data, label = zip(*data_pair)
    
    return data, label
    

checkpoint_dir = "/home/ubuntu/model/model/wifi/lstm_1228_fcnet"
ckpt_name = "lstm"
log_dir = os.path.join(checkpoint_dir, "log") 
max_epoch = 500000
batch_size = 256
learning_rate = 0.001
is_training = True

with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())
    saver, summary_writer = create_sum_and_ckpt(sess, checkpoint_dir, log_dir, ckpt_name)
    
    
    if is_training:
        for ep in range(0,max_epoch):
            
            sdata, slabel = shuffle_data(slice_data, slice_label)
            
            if ep%500 == 0 and ep !=0 and learning_rate > 0.001: learning_rate = learning_rate/2
            
            #print(sdata[0][0])
            
            for itration in range(len(slice_data)//batch_size):
                
                current_idx = itration*batch_size
                train_data = sdata[current_idx:current_idx+batch_size]
                train_label = slabel[current_idx:current_idx+batch_size]
                #train_label = np.reshape(train_label, [batch_size])
                             
                sess.run([train_op], feed_dict={inputs:train_data, labels:train_label, lr:learning_rate,dropout:1})
                #print(pre[0])
                
                
            
            if ep%1 == 0:
                
                step = ep*len(slice_data)//batch_size
                saver.save(sess,os.path.join(checkpoint_dir, ckpt_name), global_step=step)
                
              
                train_summary  = sess.run(merged_summary_train, feed_dict={inputs:train_data, labels:train_label, lr:learning_rate,dropout:1.0})
                
                testdata, testlabel = shuffle_data(testslice_data, testslice_label)
                testdata = np.stack(testdata[0:batch_size])
                testlabel = np.stack(testlabel[0:batch_size])
                #testlabel = np.reshape(testlabel, [batch_size])
                test_summary  = sess.run(merged_summary_test, feed_dict={inputs:testdata, labels:testlabel, lr:learning_rate,dropout:1.0})
                 
                summary_writer.add_summary(train_summary, ep)
                summary_writer.add_summary(test_summary, ep)
        
        
            print("Progress: {}/{}, lr:{}".format(ep+1, max_epoch, learning_rate))

    

    
    v_data = np.stack(testslice_data)
    v_label = np.stack(testslice_label)
    prediction = sess.run(tlabel, feed_dict={inputs:v_data, labels:v_label, dropout:1.0})
    calc_acc = sess.run(accuracy, feed_dict={inputs:v_data, labels:v_label, dropout:1.0})
    test_c_matrix = confusion_matrix(np.argmax(v_label, axis=1), prediction)
    
    x_axis = list(range(len(v_label)))
    plt.scatter(x_axis,np.argmax(v_label, axis=1))
    plt.show()
    plt.scatter(x_axis,prediction)
    plt.show()
    
    v_data = np.stack(slice_data)
    v_label = np.stack(slice_label)
    prediction = sess.run(tlabel, feed_dict={inputs:v_data, labels:v_label, dropout:1.0})
    calc_acc = sess.run(accuracy, feed_dict={inputs:v_data, labels:v_label, dropout:1.0})
    train_c_matrix = confusion_matrix(np.argmax(v_label, axis=1), prediction)
    
    x_axis_2 = list(range(len(v_label)))
    plt.scatter(x_axis_2,np.argmax(v_label, axis=1))
    plt.show()
    plt.scatter(x_axis_2,prediction)
    plt.show()
    
   
    
"""










