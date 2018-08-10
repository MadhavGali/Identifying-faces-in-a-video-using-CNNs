'''
Names: Geeta Madhav Gali


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

import matplotlib.pyplot as plt
import matplotlib
import theano
from array import *

# SKLEARN
from sklearn.cross_validation import train_test_split
from PIL import Image

#import the CNN model
from CNN_model import model1

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
#PeopleNames = ['Aishwarya','not Aishwarya','Saif','Unknown']
PeopleNames = ['Aishwarya', 'Priyanka']
#haarCascade instance created
#faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Function to predict wheather it is Aishwarya or not
def Predict(gray, img_rows, img_cols, nb_classes, model):#here gray = cropped face from the videoo frame
    # Create the input shape to feed the model
    X_TEST = np.zeros((1, 1, img_rows, img_cols))
    #for formatting
    # Now resize the image into (img_rows X img_cols)
    X_TEST[0, 0] = np.array(gray)
    X_TEST = X_TEST.astype('float32')
    #Normalize the input image
    X_TEST /= 255
    #Send the image for prediction to the CNN model
    value = model.predict_classes(X_TEST[0:1])
    return value

def test_video(model, font):
    #open the Testing video
    cap = cv2.VideoCapture('AshPri.mp4')
    # haarCascade instance created
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    while(cap.isOpened()):
        ret, frame = cap.read()

        if (frame != None):
            #Convert to Grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #Detect Faces
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.8,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            # if faces found, go inside the loop
            if (len(faces) > 0):
            # run a loop for each face found in the main image
                for (x, y, w, h) in faces:
                # crop the face from the main image
                    gray_cropped = gray[y:y + h, x:x + w]
                    clipped_faces = cv2.resize(gray_cropped, (img_rows, img_cols))
                    #Send the cropped face from the image(gray) to the Trained CNN model and get the predicted person's name
                    predictedPerson = Predict(clipped_faces, img_rows, img_cols, nb_classes, model)
                    print("predictedPerson = {0}".format(predictedPerson))
                    #Draw a rectangle around each face
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, PeopleNames[predictedPerson[0]], (x + w + 26, y + h), font, 1, (0, 255, 0), 2)
                    #Predict the Person
                    #Predict(gray, img_rows, img_cols, nb_classes)

                    #Show the image with the green rectangles marked
                    cv2.imshow('frame',frame)
                    #To break out of the video press "q"
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def test_create_CNN():
    #CREATE & COMPILE a CNN MODEL -----------------
    model = Sequential()
    model = model1(img_rows, img_cols, nb_classes, model, nb_filters, nb_conv, nb_pool)
    #load pre-trained weights to the CNN model
    fname = "Model1_32x32.hdf5"
    model.load_weights(fname)

    # Write some Text next to the detected face
    font = cv2.FONT_HERSHEY_SIMPLEX
    test_video(model, font)


#Everything below this comes in the main function
def main():
    test_create_CNN()






main()
