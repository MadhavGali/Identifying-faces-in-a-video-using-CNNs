'''
Names: Geeta Madhav Gali (gg6549@rit.edu)

'''

import numpy as np
import cv2
import glob
import os
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils
from sklearn.utils import shuffle


import matplotlib.pyplot as plt
import theano
from array import *

# SKLEARN
from sklearn.cross_validation import train_test_split
from PIL import Image

#import the CNN model
from CNN_model import model1

#path1 = 'C:/Users/Sanyukta/PycharmProjects/CVProject_version3/cropped faces database'
#path2 = 'C:/Users/Sanyukta/PycharmProjects/CVProject_version3/grayscale face database'
img_rows = 32
img_cols = 32
# batch_size to train
batch_size = 32
# number of output classes - Aishwarya & Not-Aishwarya
nb_classes = 2
# number of epochs to train
nb_epoch = 20
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

def resize(path1, path2, img_rows, img_cols):
    # listing is a string array which contains the names of the images
    listing = os.listdir(path1)
    print("Start Resizing")
    for file in listing:
        im = Image.open(path1 + '\\' + file)
        img = im.resize((img_rows,img_cols))
        gray = img.convert('L')
        #need to do some more processing here
        gray.save(path2 +'\\' +  file, "JPEG")
    print("Done Resizing")

def main():
    #img_rows = 32
    #img_cols = 32
    # path 1 is the path of the main image folder
    #path1 = 'database 2'#'C:/Users/Sanyukta/PycharmProjects/CVProject_version1/database 2'
    path1 = 'C:/Users/Sanyukta/PycharmProjects/CVProject_version4/cropped faces database'
    #path2 = 'gray_database/gray_database'#'C:/Users/Sanyukta/PycharmProjects/CVProject_version1/gray_database'
    path2 = 'C:/Users/Sanyukta/PycharmProjects/CVProject_version4/grayscale face database'
    #Call Resize only the very first time

    resize(path1, path2, img_rows, img_cols)
    preprocess(path1, path2)

def preprocess(path1, path2):
    #PREPROCESSING BLOCK STARTS------
    print("Starting the Preprocessing Block")
    imlist = os.listdir(path2)
    # create matrix to store all flattened images
    immatrix = np.array([np.array(Image.open(path2 + '/' + im2)).flatten() for im2 in imlist], 'f')

    #num_samples is the total number of faces
    num_samples = len(imlist)

    #label is a 1D array which is of size num_samples and has values 0 in it
    label = np.zeros((num_samples,), dtype=int)

    #all the faces in the grayscale database is assigned the label 1, which means that 1 is aishwarya
    #Aishwarya = 0, Priyanka = 1, Saif = 2
    label[0:6718] = 0
    label[6719:11619] = 1
    #label[7111:7651] = 2

    data, Label = shuffle(immatrix, label, random_state=2)
    train_data = [data, Label]

    #train_data = [immatrix, label]
    #(X, y) = (immatrix, label)
    (X, y) = (train_data[0], train_data[1])

    #split X and y into training and testing sets (validation)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

    #reshaping for CNN input layer format
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)

    #all the values are converted into float32
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    #normalizing the pixel values to be between 0 and 1, this helps the CNN converge faster
    X_train /= 255
    X_test /= 255

    # convert labels to an indicator matrix
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    print("Shape of X_train = {0}".format(X_train[0].shape))
    print("Shape of y_train = {0}".format(y_train[0]))
    print("Shape of Y_train = {0}".format(Y_train[0]))
    print("Ending the Preprocessing Block")
    # PREPROCESSING BLOCK ENDS------
    ExecuteCNN_Model(X_train, X_test, Y_train, Y_test)

def ExecuteCNN_Model(X_train, X_test, Y_train, Y_test):

    #CREATE & COMPILE a CNN MODEL -----------------
    model = Sequential()
    model = model1(img_rows, img_cols, nb_classes, model, nb_filters, nb_conv, nb_pool)

    #Load pre-trained weights if any
    fname = "Model1_32x32.hdf5"
    #While training we do not load the weights, we only save after training
    #model.load_weights(fname)

    print("Starting the Training Process .. ")
    #Train the CNN model
    hist = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
                     verbose=1, validation_data=(X_test, Y_test))

    # Store the newly trained weights
    model.save_weights(fname, overwrite=True)

    train_loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    train_acc = hist.history['acc']
    val_acc = hist.history['val_acc']
    xc = range(nb_epoch)

    plt.figure(1, figsize=(7, 5))
    plt.plot(xc, train_loss)
    plt.plot(xc, val_loss)
    plt.xlabel('num of Epochs')
    plt.ylabel('loss')
    plt.title('train_loss vs val_loss')
    plt.grid(True)
    plt.legend(['train', 'val'])
    plt.style.available  # use bmh, classic,ggplot for big pictures
    plt.style.use(['classic'])

    #Calculate and Display the Test Score
    #score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
    #print('Test score:', score[0])
    #print('Test accuracy:', score[1])

    #END OF MAIN-------------------------------------

if __name__ == "__main__":
    main()







