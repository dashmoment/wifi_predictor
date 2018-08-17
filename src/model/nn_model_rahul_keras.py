import numpy as np
import matplotlib.pyplot as plt

from keras import layers
from keras.layers import Input, Dense, Activation, BatchNormalization
from keras.layers import Dropout, Flatten
from keras.models import Model
from keras.utils import layer_utils

from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.regularizers import l1_l2

def nn_model(input_shape):

    X_input = Input(input_shape)
    X = Dense(128, kernel_initializer='he_uniform', name='fc1',
              kernel_regularizer=l1_l2(l1=0.001, l2=0.001))(X_input)
    X = Activation('relu')(X)
    X = BatchNormalization(name='bn1')(X)
    X = Dense(128, kernel_initializer='he_uniform', name='fc2',
              kernel_regularizer=l1_l2(l1=0.001, l2=0.001))(X)
    X = Activation('relu')(X)
    X = BatchNormalization(name='bn2')(X)
    X = Dense(256, kernel_initializer='he_uniform', name='fc3',
              kernel_regularizer=l1_l2(l1=0.001, l2=0.001))(X)
    X = Activation('relu')(X)
    X = BatchNormalization(name='bn3')(X)
    X = Dense(256, kernel_initializer='he_uniform', name='fc4',
              kernel_regularizer=l1_l2(l1=0.001, l2=0.001))(X)
    X = Activation('relu')(X)
    X = BatchNormalization(name='bn4')(X)
    X = Dense(512, kernel_initializer='he_uniform', name='fc5',
              kernel_regularizer=l1_l2(l1=0.001, l2=0.001))(X)
    X = Activation('relu')(X)
    X = BatchNormalization(name='bn5')(X)
    X = Dense(512, kernel_initializer='he_uniform', name='fc6',
              kernel_regularizer=l1_l2(l1=0.001, l2=0.001))(X)
    X = Activation('relu')(X)
    X = BatchNormalization(name='bn6')(X)
    X = Dense(4, activation='softmax', kernel_initializer='he_uniform',
              name='output', kernel_regularizer=l1_l2(l1=0.001, l2=0.001))(X)

    model = Model(inputs=X_input, outputs=X, name='wifi')

    return model
