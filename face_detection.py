'''
Names: Sanyukta Sanjay Kate (ssk8153@rit.edu)
       Geeta Madhav Gali (gg6549@rit.edu)

Reference: https://realpython.com/blog/python/face-recognition-with-python/
Referred from the blog given on the above link

Extra comments: Modify the folder name as per the folder in your project for opening the database and storing the
database for the cropped faces
'''

import numpy as np
import cv2
import glob
from PIL import Image

def face_detect():

    image_list = []
    count=0

    for filename in glob.glob('final_database2/*.jpg'): #assuming gif
        image_list.append(filename)

    print("length of the images list is:",len(image_list))
    #haarCascade instance created
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    #image = cv2.imread(image_list[0])
    for anImage in image_list:
        #read an image
        image = cv2.imread(anImage)
        faces = faceCascade.detectMultiScale(
            image,
            scaleFactor = 1.1,
            minNeighbors = 5,
            minSize =(30, 30),
            flags = cv2.CASCADE_SCALE_IMAGE
        )

        #if faces found, go inside the loop
        if (len(faces) > 0):
            #run a loop for each face found in the main image
            for (x, y, w, h) in faces:
                #crop the face from the main image
                crop_img = image[y:y+h, x:x+w]
                cv2.imwrite("cropped faces database/crop_img%d.jpg" % count, crop_img)
                count+=1

def main():
    face_detect()

main()


mastercard
monsento
citi bank

