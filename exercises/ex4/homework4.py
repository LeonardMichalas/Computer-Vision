#Assignment 4:

from math import hypot, pi, cos, sin
from PIL import Image
import numpy as np
import cv2 as cv
import math
 
def hough(img):

    img = im.load()
    w, h = im.size
    
    thetaAxisSize = w #Width of the hough space image
    rAxisSize = h #Height of the hough space image
    rAxisSize= int(rAxisSize/2)*2 #we make sure that this number is even

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
    maxNumbers = 9
    ignoreRadius = 10
    maxima = [0] * maxNumbers
    rhos = [0] * maxNumbers
    thetas = [0] * maxNumbers

    for u in range(0, maxNumbers):

        print('u:', u)
        value = 0 
        xposition = 0
        yposition = 0

        #find maxima in the image
        for x in range(0, w):
            for y in range(0, h):

                if(pixel_houghed_img[x,y] > value):

                    value = pixel_houghed_img[x, y]
                    xposition = x
                    yposition = y

        #Save Maxima, rhos and thetas
        maxima[u] = value
        rhos[u] = (yposition - rAxisSize/2) * d_rho
        thetas[u] = xposition * d_theta

        pixel_houghed_img[xposition, yposition] = 0

        #Delete the values around the found maxima
        radius = ignoreRadius

        for vx2 in range (-radius, radius): #checks the values around the center
            for vy2 in range (-radius, radius): #checks the values around the center
                x2 = xposition + vx2 #sets the spectated position on the shifted value 
                y2 = yposition + vy2
                
                if not(x2 < 0 or x2 >= w):
                    if not(y2 < 0 or y2 >= h):

                        pixel_houghed_img[x2, y2] = 0
                        print(pixel_houghed_img[x2, y2])
                
    print('max', maxima)
    print('rho', rhos)
    print('theta', thetas)

    return maxima, rhos, thetas

im = Image.open("img5.pgm").convert("L")
houghed_img, rAxisSize, d_rho, d_theta = hough(im)
houghed_img.save("houghspace.bmp")
houghed_img.show()

img_copy = np.ones(im.size)

maxima, rhos, thetas = find_maxima(houghed_img, rAxisSize, d_rho, d_theta)

for t in range(0, len(maxima)):
    a = math.cos(thetas[t])
    b = math.sin(thetas[t])
    x = a * rhos[t]
    y = b * rhos[t]
    pt1 = (int(x + 1000*(-b)), int(y + 1000*(a)))
    pt2 = (int(x - 1000*(-b)), int(y - 1000*(a)))
    cv.line(img_copy, pt1, pt2, (0,0,255), 3, cv.LINE_AA)
    
cv.imshow('lines', img_copy)
cv.waitKey(0)
cv.destroyAllWindows()
