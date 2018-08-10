'''
Names: Geeta Madhav Gali
'''

from keras import backend as K
K.set_image_dim_ordering('th')
from keras.models import Sequential #import the type of model
from keras.layers.core import Dense, Dropout, Activation, Flatten #import Layers
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D #import convolutional layers
from keras.utils import np_utils
from keras.layers import Input


def model1(img_rows, img_cols, nb_classes, model, nb_filters, nb_conv, nb_pool ):

    
    #Model Architecture
    #we add the input layer, 1st  layer
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            border_mode='valid',
                            input_shape=(1, img_rows, img_cols)))
    #relu-rectified linear unit activation function
    convout1 = Activation('relu')
    #add the above defined activation function
    model.add(convout1)
    # 2nd convolution layer
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    convout2 = Activation('relu')
    #we add the relu activation to the 2nd convolution layer
    model.add(convout2)

    #We add the max pooling layer
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    #model.add(Dropout(0.5))  # used for regularization to take care of overfitting

    #we flatten the max pooling output, to make it compatible with the fully connected layer
    model.add(Flatten())

    # fully connected neural network
    model.add(Dense(128))
    #relu activation to fully connected layer
    model.add(Activation('relu'))
    #model.add(Dropout(0.5))

    #output layer with two outputs - aishwarya, not-aishwarya

    model.add(Dense(nb_classes))
    #apply softmax activation
    model.add(Activation('softmax'))
    #compile the keras CNN model with adadelta optimizer
    model.compile(loss='categorical_crossentropy', optimizer='adadelta',
                  metrics=["accuracy"])  # defined our optimizer and the loss function

  

    return model