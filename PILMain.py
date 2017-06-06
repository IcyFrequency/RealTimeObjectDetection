from PIL import Image
from pylab import *
image = Image.open('/Users/pedramsherafat/PycharmProjects/RealTimeObjectDetection/imgs/attic.jpg')

im = image.convert('L')

gray()
contour(im, origin='image')

print len(array(im))
print array(im)
print array(image)
#show()