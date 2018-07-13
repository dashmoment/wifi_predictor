from sklearn.svm import SVR
import tensorflow as tf
import numpy as np
from sklearn.linear_model import Ridge


def svr_model(train_set, test_set, C=1.0, epsilon=0.3):
    
    clf = SVR(C=1.0, epsilon=0.3)
    clf.fit(train_set[0], train_set[1]) 

    prediction = clf.predict(test_set[0])
    
    return prediction

def ridge(train_set, test_set, alpha = 0.001):
    
   
    clf = Ridge(alpha=alpha)
    clf.fit(train_set[0], train_set[1]) 

    prediction = clf.predict(test_set[0])
    
    return prediction


def fc_layer(inputs, out_shape, name,initializer=tf.contrib.layers.xavier_initializer(), activat_fn=tf.nn.relu):
    
    
    pre_shape = inputs.get_shape()[-1]
    
    with tf.variable_scope(name) as scope:
        
        
        try:
            weight = tf.get_variable("weights",[pre_shape, out_shape], tf.float32, initializer=initializer)
            bias = tf.get_variable("bias",out_shape, tf.float32, initializer=initializer)
        except:
            scope.reuse_variables()
            weight = tf.get_variable("weights",[pre_shape, out_shape], tf.float32, initializer=initializer)
            bias = tf.get_variable("bias",out_shape, tf.float32, initializer=initializer)
        
        
        if activat_fn != None:
            net = activat_fn(tf.nn.xw_plus_b(inputs, weight, bias, name=name + '_out'))
        else:
            net = tf.nn.xw_plus_b(inputs, weight, bias, name=name)
        
    return net


def Spectralscan_NN(train_set, test_set, batch_size = 32, Max_epoch = 2000):
    
    input_ph = tf.placeholder(tf.float32, [None, 56], name='scans')
    label_ph = tf.placeholder(tf.float32, [None, 1], name='labels')


    with tf.name_scope("network"):
        
        net = fc_layer(input_ph, 56, "fc1")
        net = fc_layer(net, 128, "fc2")
        net = fc_layer(net, 128, "fc3")
        logits = fc_layer(net, 1, "logits", activat_fn=None)
    
    loss = tf.reduce_mean(tf.losses.absolute_difference(label_ph,logits))
    train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)
    
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    inputs = train_set[0]
    labels = train_set[1]
    
    loss_static = []
    
    for ep in range(Max_epoch):
        
        shuffle_data = list(zip(inputs, labels))
        np.random.shuffle(shuffle_data)
        inputs, labels = zip(*shuffle_data)
        
        for it in range(len(inputs)//batch_size):
            
            batch_idx = it*batch_size
            batch_input = inputs[batch_idx:batch_idx + batch_size]
            batch_labels = np.reshape(labels[batch_idx:batch_idx + batch_size],[-1,1])
        
            
            _,_loss = sess.run([train_op, loss], feed_dict={input_ph:batch_input ,label_ph:batch_labels})
            loss_static.append(loss)
            
        print("Epoch:{}, Iteration:{}, L1_loss:{}".format(ep, it, _loss))
            
    
    test_label = np.reshape(test_set[1],[-1,1]) 
    prediction = sess.run(logits, feed_dict={input_ph:test_set[0] ,label_ph:test_label})
    
    return prediction , loss_static 



def general_NN(train_set, test_set, feature = 1,batch_size = 32, Max_epoch = 2000):
    
    input_ph = tf.placeholder(tf.float32, [None, feature], name='scans')
    label_ph = tf.placeholder(tf.float32, [None, 1], name='labels')


    with tf.name_scope("network"):
        
        net = fc_layer(input_ph, feature, "fc1")
        net = fc_layer(net, 128, "fc2")
        net = fc_layer(net, 128, "fc3")
        logits = fc_layer(net, 1, "logits", activat_fn=None)
    
    loss = tf.reduce_mean(tf.losses.absolute_difference(label_ph,logits))
    train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)
    
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    inputs = train_set[0]
    labels = train_set[1]
    test_inputs = test_set[0]
    test_labels = test_set[1]
    
    loss_static = []
    
    for ep in range(Max_epoch):
        
        shuffle_data = list(zip(inputs, labels))
        np.random.shuffle(shuffle_data)
        inputs, labels = zip(*shuffle_data)
        
        for it in range(len(inputs)//batch_size):
            
            batch_idx = it*batch_size
            batch_input = inputs[batch_idx:batch_idx + batch_size]
            batch_labels = np.reshape(labels[batch_idx:batch_idx + batch_size],[-1,1])
            batch_testinput = test_inputs[0:batch_size]
            batch_testlabels = np.reshape(test_labels[0:batch_size],[-1,1])
        
            
            _,_loss = sess.run([train_op, loss], feed_dict={input_ph:batch_input ,label_ph:batch_labels})
            tloss = sess.run([loss], feed_dict={input_ph:batch_testinput ,label_ph:batch_testlabels})
            loss_static.append(loss)
            
        print("Epoch:{}, Iteration:{}, L1_loss:{}, test_loss:{}".format(ep, it, _loss, tloss))
            
    
    test_label = np.reshape(test_set[1],[-1,1]) 
    prediction = sess.run(logits, feed_dict={input_ph:test_set[0] ,label_ph:test_label})
    
    return prediction , loss_static 
            
            
            
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    
    
    