# Train and test from different set
# Add singal value
import sys
sys.path.append("../")
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.framework import ops


def initialize_parameters(layers_dim = None, initializer = tf.contrib.layers.xavier_initializer()):
    
    n_layer = len(layers_dim)
    parameters = {}
    
    for layer in range(n_layer-1):
        parameters['W' + str(layer+1)] = tf.get_variable('W' + str(layer+1), [layers_dim[layer+1], layers_dim[layer]], initializer=initializer)
        parameters['b' + str(layer+1)] = tf.get_variable('b' + str(layer+1), [layers_dim[layer+1], 1], initializer=initializer)

    return parameters



def random_mini_batches(X, Y, mini_batch_size=64):
    m = X.shape[1]   # number of training examples
    mini_batches = []
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0], m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    # number of mini batches of size mini_batch_size in your partitionning
    num_complete_minibatches = math.floor(m/mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size: k *
                                  mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size: k *
                                  mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:,
                                  num_complete_minibatches * mini_batch_size: m]
        mini_batch_Y = shuffled_Y[:,
                                  num_complete_minibatches * mini_batch_size: m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def model(X_train, Y_train, X_test, Y_test, lr=0.0001, num_epochs=1500,
          minibatch_size=32, print_cost=True):
    """
    Initializes parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [25, input_dimension]
                        b1 : [25, 1]
                        W2 : [12, 25]
                        b2 : [12, 1]
                        W3 : [output_dimension, 12]  # output_dimension = 1
                        b3 : [output_dimension, 1]
    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    """
    ops.reset_default_graph()

    ### initialize parameters ###
    # tf.set_random_seed(1)
    (n_x, m) = X_train.shape
    n_y = Y_train.shape[0]
    costs = []

    X = tf.placeholder(tf.float32, [n_x, None], name="X")
    Y = tf.placeholder(tf.float32, [n_y, None], name="Y")
    
    W1 = tf.get_variable("W1", [25, n_x], initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable("b1", [25, 1], initializer=tf.zeros_initializer())
    W2 = tf.get_variable("W2", [12, 25], initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable("b2", [12, 1], initializer=tf.zeros_initializer())
    W3 = tf.get_variable("W3", [n_y, 12], initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable("b3", [n_y, 1], initializer=tf.zeros_initializer())

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    
    ### forward propagation ###
    Z1 = tf.add(tf.matmul(W1, X), b1)
    A1 = tf.nn.relu(Z1)                     
    Z2 = tf.add(tf.matmul(W2, A1), b2)
    A2 = tf.nn.relu(Z2)                    
    Z3 = tf.add(tf.matmul(W3, A2), b3)

    ### compute cost ###
    cost = tf.reduce_mean(tf.square(Y-Z3))

    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)
    
    # initialize all the variables
    init = tf.global_variables_initializer()
    
    # start the session to compute the tensorflow graph
    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(num_epochs):
            epoch_cost = 0.
            num_minibatches = int(m / minibatch_size)
            
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size)
            
            for minibatch in minibatches:
                # select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict = {X: minibatch_X, Y:minibatch_Y})
                epoch_cost += minibatch_cost / num_minibatches
            
            if print_cost == True and epoch % 100 == 0:
                print("Cost after epoch {}: {}".format(epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
        
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title('Learning rate =' + str(lr))
        plt.savefig('../analysis_result_rahul/cost.png')
        
        # save parameters
        parameters = sess.run(parameters)
        print('Parameters have been trained')

        return parameters


def fc_layer(inputs, out_shape, name, initializer=tf.contrib.layers.xavier_initializer(), activat_fn=tf.nn.relu):
    
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


def general_NN(train_set, test_set, feature = 1,batch_size = 32, Max_epoch = 2000):
    
    input_ph = tf.placeholder(tf.float32, [None, feature], name='scans')
    label_ph = tf.placeholder(tf.float32, [None, 1], name='labels')


    with tf.name_scope("network"):
        
        net = fc_layer(input_ph, feature, "fc1")
        net = fc_layer(net, 32, "fc2")
        net = fc_layer(net, 64, "fc3")
        logits = fc_layer(net, 1, "logits", activat_fn=None)
    
    normal_loss = tf.reduce_mean(tf.square(label_ph - logits))
    reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_const = 0.4
    loss = normal_loss + reg_const * sum(reg_loss)

    learning_rate = 0.001
    #global_step = tf.Variable(0, trainable=False)
    #learning_rate = tf.train.exponential_decay(0.01, global_step, 10000, 0.96, staircase=True)
    #train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    
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
            
            
