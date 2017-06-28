import cv2
import numpy as np

image_hsv = None   # global ;(
pixel = (20,60,80) # some stupid default

# mouse callback function
def pick_color(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel = image_hsv[y,x]

        #you might want to adjust the ranges(+-10, etc):
        upper =  np.array([pixel[0] + 10, pixel[1] + 10, pixel[2] + 40])
        lower =  np.array([pixel[0] - 10, pixel[1] - 10, pixel[2] - 40])

        print pixel
        print lower
        print upper

        image_mask = cv2.inRange(image_hsv,lower,upper)
        cv2.imshow("Masking",image_mask)

def main():
    import sys
    global image_hsv, pixel # so we can use it in mouse callback
    frame = cv2.imread(sys.argv[1])  # pick.py my.png
    image_src = cv2.resize(frame, (620, 500))

    kernel = np.ones((5, 5), np.uint8)

    erosion = cv2.erode(image_src, kernel, iterations=1)
    cv2.imshow("Erosion", erosion)
    dilation = cv2.dilate(erosion, kernel, iterations=1)
    cv2.imshow("Dilation", dilation )


    if image_src is None:
        print ("the image read is None............")
        return
    cv2.imshow("Original",image_src)

    ## NEW ##
    cv2.namedWindow('HSV')
    cv2.setMouseCallback('HSV', pick_color)

    # now click into the hsv img , and look at values:
    #image_hsv = cv2.cvtColor(image_src,cv2.COLOR_BGR2HSV)
    image_hsv = cv2.cvtColor(dilation, cv2.COLOR_BGR2HSV)
    cv2.imshow("HSV",image_hsv)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()