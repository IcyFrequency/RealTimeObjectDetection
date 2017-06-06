import cv2
import os
from pylab import *

orb = cv2.ORB_create()

#folderNeg = '/Users/pedramsherafat/tf_files/polypDataTrain/neg'
#folderPos = '/Users/pedramsherafat/tf_files/polypDataTrain/pos'

folderNeg = '/Users/pedramsherafat/tf_files/test/test/0'
folderPos = '/Users/pedramsherafat/tf_files/test/test/1'

imagesNeg = []
imagesPos = []


textFile = open('/Users/pedramsherafat/PycharmProjects/RealTimeObjectDetection/FeatureTexts/gray_Test_opencv_check.txt', 'w')
textFile.write("Line1\n")
textFile.write("Line2\n")

def load_NEG_images_from_folder(folderNeg):
    #global img

    for filename in os.listdir(folderNeg):
        img = cv2.imread(os.path.join(folderNeg,filename))
        if img is not None:

            #imgF = cv2.imread(img, 0)  # queryImage        Here the features are extracted
            kp1, des1 = orb.detectAndCompute(img, None)

            # This only uses images that have an exact number of features equal to 500
            if len(des1) == 500:
                imagesNeg.append(des1)
                #print filename
                #print len(des1)
                print des1

            # For grayscale image
            #resized = cv2.resize(img, (224, 224))
            #gray_image = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            #imagesNeg.append(gray_image)


    return imagesNeg

def load_POS_images_from_folder(folderPos):

    for filename in os.listdir(folderPos):
        img = cv2.imread(os.path.join(folderPos,filename))
        if img is not None:

            #imgF = cv2.imread(img, 0)  # queryImage
            kp1, des1 = orb.detectAndCompute(img, None)
            if len(des1) == 500:
                imagesPos.append(des1)
                print filename
                print len(des1)
            # For grayscale image
            #resized = cv2.resize(img, (224, 224))
            #gray_image = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            #imagesPos.append(gray_image)
    return imagesPos

def write_NEG_features():
    for i in imagesNeg:
        textFile.write("%s " % "FileNameNeg")
        for j in i:
            if j[0] == j[0]:
                textFile.write("%s " % (j[0]))
            else:
                print 'Something went wrong NEG'
        textFile.write("0\n")
        textFile.write("\n")
        #print i

def write_POS_features():
    for a in imagesPos:
        textFile.write("%s " % "FileNamePos")
        for b in a:
            if b[0] == b[0]:
                textFile.write("%s " % (b[0]))
            else:
                print 'Something went wrong POS'
        textFile.write("1\n")
        textFile.write("\n")

print 'Extracting values from Negative folder...'
load_NEG_images_from_folder(folderNeg)
print 'Done!'
print 'Extracting values from Positive folder...'
load_POS_images_from_folder(folderPos)
print 'Done!'
print 'Writing values for negative images...'
write_NEG_features()
print 'Done!'
print 'Writing values for negative images...'
write_POS_features()
print 'Done!'
print 'Closing file!'
textFile.close()
#print "If I add %d, %d, and %d I get %d." % (my_age, my_height, my_weight, my_age + my_height + my_weight)