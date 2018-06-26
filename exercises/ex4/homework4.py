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

            treshold = 0
            col = img[x, y]
            if col <= treshold: #determines for each pixel at (x,y) if there is enough evidence of a straight line at that pixel.

                for vx in range(0, thetaAxisSize):
                    theta = d_theta * vx #angle between the x axis and the line connecting the origin with that closest point.
                    rho = x*cos(theta) + y*sin(theta) #distance from the origin to the closest point on the straight line
                    vy = rAxisSize/2 + int(rho/d_rho+0.5) #Berechne Y-Werte im hough space image
                    pixel_houghed_img[vx, vy] += 1 #voting

    return houghed_img, rAxisSize, d_rho, d_theta

def find_maxima(houghed_img, rAxisSize, d_rho, d_theta):

    w, h = houghed_img.size
    pixel_houghed_img = houghed_img.load()
    maxNumbers = 10
    maxima = [0] * maxNumbers
    rhos = [0] * maxNumbers
    thetas = [0] * maxNumbers
    max1, max2, max3, max4, max5, max6, max7, max8, max9, max10, max11 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    rho1, rho2, rho3, rho4, rho5, rho6, rho7, rho8, rho9, rho10, rho11 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    theta1, theta2, theta3, theta4, theta5, theta6, theta7, theta8, theta9, theta10, theta11 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    
    for x in range(1, w):
        for y in range(1, h):
            value = pixel_houghed_img[x, y]

            if(value > max1):

                max1 = value
                rho1 = (y - rAxisSize/2) * d_rho
                theta1 = x * d_theta
            
            elif(value > max2):
                
                max2 = value
                rho2 = (y - rAxisSize/2) * d_rho
                theta2 = x * d_theta
                
            elif(value > max3):

                max3 = value
                rho3 = (y - rAxisSize/2) * d_rho
                theta3 = x * d_theta

            elif(value > max4):

                max4 = value
                rho4 = (y - rAxisSize/2) * d_rho
                theta4 = x * d_theta    
            
            elif(value > max5):

                max5 = value
                rho5 = (y - rAxisSize/2) * d_rho
                theta5 = x * d_theta

            elif(value > max6):

                max6 = value
                rho6 = (y - rAxisSize/2) * d_rho
                theta6 = x * d_theta    
            
            elif(value > max7):

                max7 = value
                rho7 = (y - rAxisSize/2) * d_rho
                theta7 = x * d_theta

            elif(value > max8):

                max8 = value
                rho8 = (y - rAxisSize/2) * d_rho
                theta8 = x * d_theta  

            elif(value > max9):

                max9 = value
                rho9 = (y - rAxisSize/2) * d_rho
                theta9 = x * d_theta

            elif(value > max10):

                max10 = value
                rho10 = (y - rAxisSize/2) * d_rho
                theta10 = x * d_theta                      
                    
    print('max', max1, max2, max3, max4, max5, max6, max7, max8, max9, max10, max11)
    print('rho', rho1, rho2, rho3, rho4, rho5, rho6, rho7, rho8, rho9, rho10, rho11)
    print('theta', theta1, theta2, theta3, theta4, theta5, theta6, theta7, theta8, theta9, theta10, theta11)

    # Results of the print:
    # ('max', 155, 144, 142, 119, 119, 104, 103, 98)
    # ('rho', 120, 264, 157, 121, 119, 198, 197, 197)
    # ('theta', 416, 31, 458, 414, 417, 288, 291, 292)
    return rho1, rho2, rho3, rho4, rho5, rho6, rho7, rho8, rho9, rho10, rho11, theta1, theta2, theta3, theta4, theta5, theta6, theta7, theta8, theta9, theta10, theta11 

im = Image.open("img3.pgm").convert("L")
houghed_img, rAxisSize, d_rho, d_theta = hough(im)
houghed_img.save("ho.bmp")
houghed_img.show()

img_copy = np.ones(im.size)

rho1, rho2, rho3, rho4, rho5, rho6, rho7, rho8, rho9, rho10, rho11, theta1, theta2, theta3, theta4, theta5, theta6, theta7, theta8, theta9, theta10, theta11 = find_maxima(houghed_img, rAxisSize, d_rho, d_theta)

a1 = math.cos(theta1)
b1 = math.sin(theta1)
x01 = a1 * rho1
y01 = b1 * rho1
pt11 = (int(x01 + 1000*(-b1)), int(y01 + 1000*(a1)))
pt21 = (int(x01 - 1000*(-b1)), int(y01 - 1000*(a1)))
cv.line(img_copy, pt11, pt21, (0,0,255), 3, cv.LINE_AA)

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

a4 = math.cos(theta3)
b4 = math.sin(theta3)
x04 = a4 * rho4
y04 = b4 * rho4
pt14 = (int(x04 + 1000*(-b4)), int(y04 + 1000*(a4)))
pt24 = (int(x04 - 1000*(-b4)), int(y04 - 1000*(a4)))
cv.line(img_copy, pt14, pt24, (0,0,255), 3, cv.LINE_AA)

a5 = math.cos(theta5)
b5 = math.sin(theta5)
x05 = a5 * rho5
y05 = b5 * rho5
pt15 = (int(x05 + 1000*(-b5)), int(y05 + 1000*(a5)))
pt25 = (int(x05 - 1000*(-b5)), int(y05 - 1000*(a5)))
cv.line(img_copy, pt15, pt25, (0,0,255), 3, cv.LINE_AA)

a6 = math.cos(theta6)
b6 = math.sin(theta6)
x06 = a6 * rho6
y06 = b6 * rho6
pt16 = (int(x06 + 1000*(-b6)), int(y06 + 1000*(a6)))
pt26 = (int(x06 - 1000*(-b6)), int(y06 - 1000*(a6)))
cv.line(img_copy, pt16, pt26, (0,0,255), 3, cv.LINE_AA)

a7 = math.cos(theta3)
b7 = math.sin(theta3)
x07 = a7 * rho7
y07 = b7 * rho7
pt17 = (int(x07 + 1000*(-b7)), int(y07 + 1000*(a7)))
pt27 = (int(x07 - 1000*(-b7)), int(y07 - 1000*(a7)))
cv.line(img_copy, pt17, pt27, (0,0,255), 3, cv.LINE_AA)

a8 = math.cos(theta8)
b8 = math.sin(theta8)
x08 = a8 * rho8
y08 = b8 * rho8
pt18 = (int(x08 + 1000*(-b8)), int(y08 + 1000*(a8)))
pt28 = (int(x08 - 1000*(-b8)), int(y08 - 1000*(a8)))
cv.line(img_copy, pt18, pt18, (0,0,255), 3, cv.LINE_AA)

a9 = math.cos(theta9)
b9 = math.sin(theta9)
x09 = a9 * rho9
y09 = b9 * rho9
pt19 = (int(x09 + 1000*(-b9)), int(y09 + 1000*(a9)))
pt29 = (int(x09 - 1000*(-b9)), int(y09 - 1000*(a9)))
cv.line(img_copy, pt19, pt19, (0,0,255), 3, cv.LINE_AA)

a10 = math.cos(theta10)
b10 = math.sin(theta10)
x10 = a10 * rho10
y10 = b10 * rho10
pt110 = (int(x10 + 1000*(-b10)), int(y10 + 1000*(a10)))
pt210 = (int(x10 - 1000*(-b10)), int(y10 - 1000*(a10)))
cv.line(img_copy, pt110, pt110, (0,0,255), 3, cv.LINE_AA)

a11 = math.cos(theta11)
b11 = math.sin(theta11)
x11 = a11 * rho11
y11 = b11 * rho11
pt111 = (int(x11 + 1000*(-b11)), int(y11 + 1000*(a11)))
pt211 = (int(x11 - 1000*(-b11)), int(y11 - 1000*(a11)))
cv.line(img_copy, pt111, pt111, (0,0,255), 3, cv.LINE_AA)

cv.imshow('lines', img_copy)
cv.waitKey(0)
cv.destroyAllWindows()
