#Assignment 4:

from math import hypot, pi, cos, sin
from PIL import Image
import numpy as np
import cv2 as cv
import math
 
def hough(img):
    
    thetaAxisSize = 460 #Width of the hough space image
    rAxisSize = 360 #Height of the hough space image
    rAxisSize= int(rAxisSize/2)*2 #we make sure that this number is even

    img = im.load()
    w, h = im.size

    houghed_img = Image.new("L", (thetaAxisSize, rAxisSize), 0) #legt Bildgroesse fest
    pixel_houghed_img = houghed_img.load()

    max_radius = hypot(w, h)
    d_theta = pi / thetaAxisSize
    d_rho = max_radius / (rAxisSize/2) 
  
 
    #Accumulator
    for x in range(0, w):
        for y in range(0, h):

            treshold = 255
            col = img[x, y]
            if col >= treshold: #determines for each pixel at (x,y) if there is enough evidence of a straight line at that pixel.

                for vx in range(0, thetaAxisSize):
                    theta = d_theta * vx #angle between the x axis and the line connecting the origin with that closest point.
                    rho = x*cos(theta) + y*sin(theta) #distance from the origin to the closest point on the straight line
                    vy = rAxisSize/2 + int(rho/d_rho+0.5) #Berechne Y-Werte im hough space image
                    pixel_houghed_img[vx, vy] += 1 #voting

    return houghed_img

def find_maxima(houghed_img):

    w, h = houghed_img.size
    pixel_houghed_img = houghed_img.load()
    max1, max2, max3 = 0, 0, 0
    rho1, rho2, rho3 = 0, 0, 0
    theta1, theta2, theta3 = 0, 0, 0
    
    for x in range(1, w):
        for y in range(1, h):
            value = pixel_houghed_img[x, y]

            if(value > max1):

                max1 = value
                x1position = x
                y1position = y
                rho1 = y
                theta1 = x
            '''
            elif(value > max2):
                
                max2 = value
                x2position = x
                x3position = y
                rho2 = y
                theta2 = x
                
            elif(value > max3):

                max3 = value
                x3position = x
                y3position = y
                rho3 = y
                theta3 = x
            '''
    print('max', max1, max2, max3)
    print('rho', rho1, rho2, rho3)
    print('theta', theta1, theta2, theta3)

    # Results of the print:
    # ('max', 255, 255, 255)
    # ('rho', 1, 1, 1)
    # ('theta', 183, 184, 186)
    return rho1, theta1, rho2, theta2, rho3, theta3       

im = Image.open("img5.pgm").convert("L")
houghed_img = hough(im)
houghed_img.save("ho.bmp")
houghed_img.show()

img_copy = np.ones(im.size)

rho1, theta1, rho2, theta2, rho3, theta3 = find_maxima(houghed_img)

a1 = math.cos(theta1)
b1 = math.sin(theta1)
x01 = a1 * rho1
y01 = b1 * rho1
pt11 = (int(x01 + 1000*(-b1)), int(y01 + 1000*(a1)))
pt21 = (int(x01 - 1000*(-b1)), int(y01 - 1000*(a1)))
cv.line(img_copy, pt11, pt21, (0,0,255), 3, cv.LINE_AA)
'''
a2 = math.cos(theta2)
b2 = math.sin(theta2)
x02 = a2 * rho2
y02 = b2 * rho2
pt12 = (int(x02 + 1000*(-b2)), int(y02 + 1000*(a2)))
pt22 = (int(x02 - 1000*(-b2)), int(y02 - 1000*(a2)))
cv.line(img_copy, pt12, pt22, (0,0,255), 3, cv.LINE_AA)

a3 = math.cos(theta3)
b3 = math.sin(theta3)
x03 = a3 * rho3
y03 = b3 * rho3
pt13 = (int(x03 + 1000*(-b3)), int(y03 + 1000*(a3)))
pt23 = (int(x03 - 1000*(-b3)), int(y03 - 1000*(a3)))
cv.line(img_copy, pt13, pt23, (0,0,255), 3, cv.LINE_AA)
'''
cv.imshow('lines', img_copy)
cv.waitKey(0)
cv.destroyAllWindows()
