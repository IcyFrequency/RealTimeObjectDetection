from PIL import Image
import pickle
import threading
from keras.preprocessing import image as image_utils
import argparse
import cv2
import numpy as np
import os
import random
import sys
from pylab import *
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier

frame = None
model_path = '/Users/pedramsherafat/PycharmProjects/RealTimeObjectDetection/Models/Classifier_KNeighborsClassifier.pkl'

model = pickle.load(open(model_path, 'rb'))

orb = cv2.ORB_create()

'''
img_path = ('/Users/pedramsherafat/PycharmProjects/RealTimeObjectDetection/imgs/attic.jpg')
img = image_utils.load_img(img_path, target_size=(224, 224))
'''

class MyThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

#   Read frame from camera
cap = cv2.VideoCapture(0)

#   Check camera status
if (cap.isOpened()):
    print("Camera OK")
else:
    cap.open()
temp_array = []

while (True):
    #   Read frame
    ret, original = cap.read()
    #   Resize frame
    frame = cv2.resize(original, (1280, 720))
    #   Extract features from frame
    #gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp1, des1 = orb.detectAndCompute(frame, None)

    print des1
    model.predict(des1)



    #   Classify frame
    """
    # Iterate through des1 array and extract only feature values without type into seperate array x
    for i in des1:
        #textFile.write("%s " % "FileNameNeg")
        for j in i:
            if j[0] == j[0]:
                #textFile.write("%s " % (j[0]))
                temp_array.append(j[0])
            else:
                print 'Something went wrong NEG'


    # Make prediction on the des1 feature values from the new array x
    if len(temp_array) == 500:
        print temp_array
        print len(temp_array)
        model.predict(temp_array)
    """
    #   Print out feature values
    #print array(gray_image)

    cv2.imshow("Classification", frame)

    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break;

cap.release()
frame = None
cv2.destroyAllWindows()
sys.exit()