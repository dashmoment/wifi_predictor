import numpy as np
import matplotlib.pyplot as plt

from keras import layers
from keras.layers import Input, Dense, Activation, BatchNormalization
from keras.layers import Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.utils import layer_utils

from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.regularizers import l1_l2


def nn_model(input_shape):

    X_input = Input(input_shape)
    X = Dense(128, kernel_initializer='he_normal', name='fc1',
              kernel_regularizer=l1_l2(l1=0.0, l2=0.1))(X_input)
    X = BatchNormalization(name='bn1')(X)
    X = Activation('relu')(X)
    #X = Dropout(0.2)(X)
    X = Dense(128, kernel_initializer='he_normal', name='fc2',
              kernel_regularizer=l1_l2(l1=0.0, l2=0.1))(X)
    X = BatchNormalization(name='bn2')(X)
    X = Activation('relu')(X)
    #X = Dropout(0.2)(X)
    X = Dense(256, kernel_initializer='he_normal', name='fc3',
              kernel_regularizer=l1_l2(l1=0.0, l2=0.1))(X)
    X = BatchNormalization(name='bn3')(X)
    X = Activation('relu')(X)
    #X = Dropout(0.2)(X)
    X = Dense(256, kernel_initializer='he_normal', name='fc4',
              kernel_regularizer=l1_l2(l1=0.0, l2=0.1))(X)
    X = BatchNormalization(name='bn4')(X)
    X = Activation('relu')(X)
    #X = Dropout(0.2)(X)
    
    X = Dense(512, kernel_initializer='he_normal', name='fc5',
              kernel_regularizer=l1_l2(l1=0.0, l2=0.1))(X)
    X = BatchNormalization(name='bn5')(X)
    X = Activation('relu')(X)
    #X = Dropout(0.2)(X)
    X = Dense(512, kernel_initializer='he_normal', name='fc6',
              kernel_regularizer=l1_l2(l1=0.0, l2=0.1))(X)
    X = BatchNormalization(name='bn6')(X)
    X = Activation('relu')(X)
    #X = Dropout(0.2)(X)
    '''
    X = Dense(64, kernel_initializer='he_uniform', name='fc7',
              kernel_regularizer=l1_l2(l1=0.000, l2=0.0000))(X)
    X = BatchNormalization(name='bn7')(X)
    X = Activation('relu')(X)
    X = Dropout(0.1)(X)
    X = Dense(64, kernel_initializer='he_uniform', name='fc8',
              kernel_regularizer=l1_l2(l1=0.000, l2=0.0000))(X)
    X = BatchNormalization(name='bn8')(X)
    X = Activation('relu')(X)
    X = Dropout(0.1)(X)
    
    X = Dense(128, kernel_initializer='he_uniform', name='fc9',
              kernel_regularizer=l1_l2(l1=0.000, l2=0.0000))(X)
    X = BatchNormalization(name='bn9')(X)
    X = Activation('relu')(X)
    X = Dropout(0.1)(X)
    X = Dense(128, kernel_initializer='he_uniform', name='fc10',
              kernel_regularizer=l1_l2(l1=0.000, l2=0.0000))(X)
    X = BatchNormalization(name='bn10')(X)
    X = Activation('relu')(X)
    X = Dropout(0.1)(X)
    '''
    X = Dense(4, activation='softmax', kernel_initializer='he_normal',
              name='output', kernel_regularizer=l1_l2(l1=0.0, l2=0.1))(X)

    model = Model(inputs=X_input, outputs=X, name='wifi')

    return model


'''
def nn_model(input_shape):

    X_input = Input(input_shape)
    X = Dense(128, kernel_initializer='he_uniform', name='fc1',
              kernel_regularizer=l1_l2(l1=0.000, l2=0.0001))(X_input)
    X = BatchNormalization(name='bn1')(X)
    X = LeakyReLU(alpha=0.1)(X)
    X = Dense(128, kernel_initializer='he_uniform', name='fc2',
              kernel_regularizer=l1_l2(l1=0.000, l2=0.0001))(X)
    X = BatchNormalization(name='bn2')(X)
    X = LeakyReLU(alpha=0.1)(X)
    X = Dense(256, kernel_initializer='he_uniform', name='fc3',
              kernel_regularizer=l1_l2(l1=0.000, l2=0.0001))(X)
    X = BatchNormalization(name='bn3')(X)
    X = LeakyReLU(alpha=0.1)(X)
    X = Dense(256, kernel_initializer='he_uniform', name='fc4',
              kernel_regularizer=l1_l2(l1=0.000, l2=0.0001))(X)
    X = BatchNormalization(name='bn4')(X)
    X = LeakyReLU(alpha=0.1)(X)
    X = Dense(512, kernel_initializer='he_uniform', name='fc5',
              kernel_regularizer=l1_l2(l1=0.000, l2=0.0001))(X)
    X = BatchNormalization(name='bn5')(X)
    X = LeakyReLU(alpha=0.1)(X)
    X = Dense(512, kernel_initializer='he_uniform', name='fc6',
              kernel_regularizer=l1_l2(l1=0.000, l2=0.0001))(X)
    X = BatchNormalization(name='bn6')(X)
    X = LeakyReLU(alpha=0.1)(X)
    X = Dense(4, activation='softmax', kernel_initializer='he_uniform',
              name='output', kernel_regularizer=l1_l2(l1=0.000, l2=0.0001))(X)

    model = Model(inputs=X_input, outputs=X, name='wifi')

    return model


'''
