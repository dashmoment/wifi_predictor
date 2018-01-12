
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import os
import random
#import model_zoo as mz

#import tensorflow.contrib.eager as tfe
from matplotlib import pyplot as plt



### Data Prerpocess

# Training data


h5data = pd.HDFStore('../data/ProcessData1213/training_data_v2.h5')
raw_data= h5data["raw_data"]

label_feature = 'FER'
slice_data = raw_data['10:144:24:24']['sigval_std']
slice_label = raw_data['10:144:24:24'][label_feature]

h5data = pd.HDFStore('../data/ProcessData1213/test_data_v2.h5')
raw_data= h5data["raw_data"]
testslice_data = raw_data['10:144:24:24']['sigval_std']
testslice_label = raw_data['10:144:24:24'][label_feature]

 

############## Build Model###################
feature_size = len(slice_data[0][0])
time_step = len(slice_data[0])
inputs = tf.placeholder(tf.float32, [None, time_step, feature_size])
labels = tf.placeholder(tf.float32, [None, 1])
lr = tf.placeholder(tf.float32, name='learning_rate')




num_hidden = 64
num_classes = 1

weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}

with tf.variable_scope("lstm", reuse=None):

    ustack_in = tf.unstack(inputs, time_step, 1)
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    outputs, states = rnn.static_rnn(lstm_cell, ustack_in, dtype=tf.float32)
    logits = tf.matmul(outputs[-1], weights['out']) + biases['out']

### aux function
    
loss = l1loss = tf.reduce_mean(tf.losses.absolute_difference(labels,logits))
mse =  tf.reduce_mean(tf.squared_difference(labels,logits))   
train_op = tf.train.AdamOptimizer(lr).minimize(loss)


with tf.name_scope('train_summary'):
    
    tf.summary.scalar("l1_loss", l1loss, collections=['train'])
    tf.summary.scalar("mse", mse, collections=['train'])
    
    merged_summary_train = tf.summary.merge_all('train')  
    
with tf.name_scope('test_summary'):
    
    tf.summary.scalar("l1_loss", l1loss, collections=['test'])
    tf.summary.scalar("mse", mse, collections=['test'])
    
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
    

checkpoint_dir = "/home/ubuntu/model/model/wifi/lstm_1213_FER_dropout"
ckpt_name = "lstm"
log_dir = os.path.join(checkpoint_dir, "log") 
max_epoch = 10000
batch_size = 64
learning_rate = 1e-4
is_training = False

with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())
    saver, summary_writer = create_sum_and_ckpt(sess, checkpoint_dir, log_dir, ckpt_name)
    
    
    if is_training:
        for ep in range(max_epoch):
            
            sdata, slabel = shuffle_data(slice_data, slice_label)
            
            if ep%2000 and ep !=0: learning_rate = learning_rate
            
            for itration in range(len(slice_data)//batch_size):
                
                current_idx = itration*batch_size
                train_data = np.stack(sdata[current_idx:current_idx+batch_size])
                train_label = np.stack(slabel[current_idx:current_idx+batch_size])
                train_label = np.reshape(train_label, [batch_size,1])
                
                sess.run(train_op, feed_dict={inputs:train_data, labels:train_label, lr:learning_rate})
               
            
            
            if ep%50 == 0:
                
                step = ep*len(slice_data)//batch_size
                saver.save(sess,os.path.join(checkpoint_dir, ckpt_name), global_step=step)
                
              
                train_summary  = sess.run(merged_summary_train, feed_dict={inputs:train_data, labels:train_label})
                
                testdata, testlabel = shuffle_data(testslice_data, testslice_label)
                testdata = np.stack(testdata[0:batch_size])
                testlabel = np.stack(testlabel[0:batch_size])
                testlabel = np.reshape(testlabel, [batch_size,1])
                test_summary  = sess.run(merged_summary_test, feed_dict={inputs:testdata, labels:testlabel})
                 
                summary_writer.add_summary(train_summary, ep)
                summary_writer.add_summary(test_summary, ep)
        
        
            print("Progress: {}/{}".format(ep+1, max_epoch))

    
    
    
    v_data = np.stack(testslice_data)
    v_label = np.stack(testslice_label)
    prediction = sess.run(logits, feed_dict={inputs:v_data})
    
    # Plot result and labbel
    
    
    x_axis = list(range(len(prediction)))
    plt.plot(x_axis, v_label)
    plt.show()
    plt.plot(x_axis, prediction)
    plt.show()
    plt.plot(x_axis, v_label)
    plt.plot(x_axis, prediction)
    plt.show()
    
    prediction = np.reshape(prediction, [-1])
    print("Test L1 Mean: ", np.mean(abs(v_label-prediction)))
    print("Test L1 std: ", np.std(abs(v_label-prediction)))
    
    counts, bin_edges = np.histogram(abs(v_label-prediction), bins=20, normed=True)
    cdf = np.cumsum(counts)
    plt.plot(bin_edges[1:], cdf)
    plt.show()
    


    v_data = np.stack(slice_data)
    v_label = np.stack(slice_label)
    prediction = sess.run(logits, feed_dict={inputs:v_data})
    
    x_axis = list(range(len(prediction)))
    plt.plot(x_axis, v_label)
    plt.show()
    plt.plot(x_axis, prediction)
    plt.show()
    plt.plot(x_axis, v_label)
    plt.plot(x_axis, prediction)
    plt.show()
    
    print("Train L1 Mean: ", np.mean(abs(v_label-prediction)))
    print("Train L1 std: ", np.std(abs(v_label-prediction)))













