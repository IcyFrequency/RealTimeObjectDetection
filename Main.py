import cv2
import numpy as np
from matplotlib import pyplot as plt

from sklearn.feature_extraction import image



img1 = cv2.imread('/Users/pedramsherafat/PycharmProjects/RealTimeObjectDetection/imgs/attic.jpg')

patches = image.extract_patches_2d(img1, (2, 2), max_patches=2,
    random_state=0)
patches.shape

print patches

plt.subplot(),plt.imshow(img1,'gray'),plt.title('ORIGINAL')
plt.show()