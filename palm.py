import cv2
import numpy as np

hand = "left"

for i in range(10,11):
    print i
    img = cv2.imread(hand + 'Edited/' + hand + str(i) + '.jpg')

    # noise
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    noise = cv2.fastNlMeansDenoising(gray)
    noise = cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR)
    #cv2.imwrite("img/noise.jpg", noise)

    # equalist hist
    kernel = np.ones((7,7),np.uint8)
    img = cv2.morphologyEx(noise, cv2.MORPH_OPEN, kernel)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    #cv2.imwrite("img/equalizeHist.jpg", img_output)

    # invert
    inv = cv2.bitwise_not(img_output)
    #cv2.imwrite("img/inverted.jpg", inv)

    # erode
    gray = cv2.cvtColor(inv, cv2.COLOR_BGR2GRAY)
    size = np.size(gray)
    erosion = cv2.erode(gray,kernel,iterations = 1)
    #cv2.imwrite("img/eroded.jpg", erosion)

    # skel
    img = gray.copy() # don't clobber original
    skel = img.copy()
    skel[:,:] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))
    iterations = 0

    while True:
        eroded = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
        temp = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, kernel)
        temp  = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img[:,:] = eroded[:,:]
        if cv2.countNonZero(img) == 0:
            break

    ret, thr = cv2.threshold(skel, 5,255, cv2.THRESH_BINARY);

    #cv2.imwrite("img/skel.jpg", skel)
    cv2.imwrite(hand + "Edited/thr" + str(i) + ".jpg", thr)

    # SIFT
    #sift = cv2.xfeatures2d.SURF_create()
    #kp = sift.detect(thr,None)
    #siftImg = cv2.drawKeypoints(thr,kp, thr)

    #cv2.imwrite('img/sift.jpg',siftImg)

