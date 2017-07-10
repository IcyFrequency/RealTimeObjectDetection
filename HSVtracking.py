import cv2
import numpy as np
import time

"""
original = cv2.imread('/Users/pedramsherafat/PycharmProjects/RealTimeObjectDetection/imgs/Adenomatous.jpg')
frame = cv2.resize(original, (400, 400))
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

lower_red = np.array([30, 150, 50])
upper_red = np.array([255, 255, 180])

mask = cv2.inRange(hsv, lower_red, upper_red)
res = cv2.bitwise_and(frame, frame, mask=mask)

kernel = np.ones((5, 5), np.uint8)

opening = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

cv2.imshow('Original', frame)
cv2.imshow('Mask', mask)
cv2.imshow('Opening', opening)
cv2.imshow('Closing', closing)

cv2.waitKey()

"""
cap = cv2.VideoCapture(0)
startMeasure = time.time()
while (1):
    startTime = time.time()
    _, original = cap.read()

    frame = cv2.resize(original, (400, 400))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red = np.array([30, 150, 50])
    upper_red = np.array([255, 255, 180])

    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    kernel = np.ones((5, 5), np.uint8)

    opening = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    cv2.imshow('Original', frame)
    cv2.imshow('Mask', mask)
    cv2.imshow('Opening', opening)
    cv2.imshow('Closing', closing)

    elapsedTime = time.time() - startTime
    elapsedTotTime = (time.time() - startMeasure)
    #print('function finished in {} ms'.format(float(elapsedTime * 1000)))
    print('{} \t %f '.format(float(elapsedTime * 1000)) % elapsedTotTime)
    #rint "time.time(): %f " % time.time()


    #print ('FPS: {}'.format(int(1000/(elapsedTime))))
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

TotalTime = time.time() - startMeasure
print('Total time: {}'.format(float(TotalTime * 1000)))
cv2.destroyAllWindows()
cap.release()

""" Read from folder


from os import listdir
from os.path import isfile, join
import numpy
import cv2

mypath='/path/to/folder'
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
images = numpy.empty(len(onlyfiles), dtype=object)
for n in range(0, len(onlyfiles)):
  images[n] = cv2.imread( join(mypath,onlyfiles[n]) )

"""
