import cv2
import numpy as np
import time

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

while (1):

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

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
"""