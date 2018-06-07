#Assignment 2:
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from PIL import Image
import scipy.ndimage as ndimage
from scipy.ndimage.filters import gaussian_filter


# 1. Load Image
name = 'ebay.png'
img = cv.imread(name) #import image
h, w = img.shape[:2]
kernel = 5
radius = (kernel-1)/2

img2 = np.zeros((h, w, 3), dtype = 'uint8') #new image to paint on

def pxIsInImgRange(x, y):
    if (0<=x) and (x < w): #wenn es bei 0/0 ist und kleiner als w
            if (0<=y) and (y < h): #sind im Bild bereich
                return True
    return False


for x in range (-radius, w+radius):
    for y in range (-radius, h+radius):

        if pxIsInImgRange(x,y): #prueft ob aktuller position im Bereich, erst dann starten wir algorithmus
                
                px = 0
                
                for vx2 in range (-radius, radius+1): #um den mittleren pixel herum
                    for vy2 in range (-radius, radius+1):
                        x2 = x + vx2
                        y2 = y + vy2
                        if pxIsInImgRange(x2,y2): #prueft ob die aktulle verschiebung im Bildbereich, sonst ist der ausserhalb liegende bereich 0/schwarz
                            px = px + (img[y2][x2]/float((kernel*kernel)))
                        else:
                            px = px + 0 #nichts auf erdieren, falls aktuelle verschiebung ausserhalb des bildes

                #Nach den beiden vorschleifen, die aufsummierten durchschnitte in das zenter des patches
                img2[y][x] = px

                
new_image = Image.fromarray(img2)
new_image.save('new.png')

# 4. Show image

new_image.show()





"""
kernel = 9
half_kernel = (kernel-1)/2
img2 = np.zeros((h+half_kernel,w+half_kernel,3), dtype = 'uint8')
print(img2[0][0])

for y in range (0, w+kernel):
    for x in range (0, h+kernel):
        x2 = x-(kernel-1)/2
        y2 = y-(kernel-1)/2
        if (0 <= x2 and x < h) and (0 <= y2 and y < w):
            img2[x][y] = img[x][y]

#smoothedImage = cv.imread(name) #initialize new image

# 2. Smooth with with kernel size 3

def smooth(kernel, img2):

    for y in range(0, (w-(kernel-1)/2)):
        for x in range(0, (h-(kernel-1)/2)):
            px = 0
            for i in range (0, kernel):
                for j in range (0, kernel):     
                    px = px + (img2[x+i][y+j]/float((kernel*kernel))) #0/0
            
            img2[x+((kernel-1)/2)][y+((kernel-1)/2)] = px   #1/1
    return img2

'''   
        for y in range(0, w-4):
            for x in range(0, h-4):     
                px1 = img[x][y] #0/0
                px2 = img[x][y+1] #0/1
                px3 = img[x][y+2] #0/2
                px4 = img[x][y+3] #1/0
                px5 = img[x][y+4] #1/1
                px6 = img[x+1][y] #1/2
                px7 = img[x+1][y+1]#2/0
                px8 = img[x+1][y+2] #2/1
                px9 = img[x+1][y+3] #2/2
                px10 = img[x+1][y+4] #0/0
                px11 = img[x+2][y] #0/1
                px12 = img[x+2][y+1] #0/2
                px13 = img[x+2][y+2] #1/0
                px14 = img[x+2][y+3] #1/1
                px15 = img[x+2][y+4] #1/2
                px16 = img[x+3][y] #2/0
                px17 = img[x+3][y+1] #2/1
                px18 = img[x+3][y+2] #2/2
                px19 = img[x+3][y+3] #0/0
                px20 = img[x+3][y+4] #0/1
                px21 = img[x+4][y] #0/2
                px22 = img[x+4][y+1] #1/0
                px23 = img[x+4][y+2] #1/1
                px24 = img[x+4][y+3] #1/2
                px25 = img[x+4][y+4] #2/0

                average = px1/9. + px2/9. + px3/9. + px4/9. + px5/9. + px6/9. + px7/9. + px8/9. + px9/9.
                
                smoothedImage[x+2][y+2] = average      
    '''

# 3. Transform the resulting image into pgm format and save result        

smoothedImage = smooth(kernel, img2)
"""


