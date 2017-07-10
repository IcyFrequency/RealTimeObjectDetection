import cv2
import numpy as np

# img = cv2.imread('/Users/pedramsherafat/PycharmProjects/RealTimeObjectDetection/imgs/Adenomatous.png')
# kernel = np.ones((5,5),np.uint8)
# erosion = cv2.erode(img,kernel,iterations = 1)

# cv2.imshow("Suvage", erosion)
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while (1):

    _, frame = cap.read()

    # frame = cv2.imread('/Users/pedramsherafat/PycharmProjects/RealTimeObjectDetection/imgs/Adenomatous.png')
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # lower_red = np.array([30, 150, 50])
    # upper_red = np.array([255, 255, 180])

    # mask = cv2.inRange(hsv, lower_red, upper_red)
    # res = cv2.bitwise_and(frame, frame, mask=mask)

    kernel = np.ones((5, 5), np.uint8)

    erosion = cv2.erode(frame, kernel, iterations=1)
    dilation = cv2.dilate(erosion, kernel, iterations=1)
    erosionHSV = cv2.erode(hsv, kernel, iterations=1)
    dilationHSV = cv2.dilate(erosionHSV, kernel, iterations=1)

    cv2.imshow('Original', frame)
    cv2.imshow('HSV', hsv)
    cv2.imshow('Erosion', erosion)
    cv2.imshow('ErosionHSV', erosionHSV)
    cv2.imshow('Dialation', dilation)
    cv2.imshow('DialationHSV', dilationHSV)
    # cv2.imshow('Dilation', dilation)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()

cv2.waitKey()
