
image = '/Users/pedramsherafat/PycharmProjects/RealTimeObjectDetection/imgs/zelda2.jpg'
image2 = '/Users/pedramsherafat/PycharmProjects/RealTimeObjectDetection/imgs/zelda.jpg'

import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread(image, 0)  # queryImage
img2 = cv2.imread(image2, 0)  # trainImage

# Initiate ORB detector
orb = cv2.ORB_create()

# Initiate SIFT detector
#sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

#kp11, des11 = sift.detectAndCompute(img1, None)
#kp22, des22 = sift.detectAndCompute(img2, None)

print des1
print des2
print len(des1)
print len(des2)

#print des11
#print des22
#print len(des11)
#print len(des22)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1, des2)  # Sort them in the order of their distance.
matches = sorted(matches, key=lambda x: x.distance)

# Draw first 10 matches.
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2)

plt.imshow(img3), plt.show()

