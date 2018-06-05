#Assignment 2:
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

# 1. Load Image
name = 'ebay.png'
img = cv.imread(name) #import image

h = img.shape[0] #number of pixels in the hight
w = img.shape[1] #number of piexels in the weight

pixelList = [] #array which later can save the pixel values of the image

print(h)
print(w)

# 2. Prepare pixel array

for y in range(0, w):
    for x in range(0, h):       
        px = img[x][y] #reads the pixel which is a npndarray [][][]
        pixelList.append(px) #saves the pixel data of every pixel we loop so we can use it later to plot the histogram

newImage = np.array(pixelList)  

# 2. Smooth with a flexibel kernel (size 3, 5, 7, 9)

# 3. Transform the resulting image into pgm format

# 4. Save result

# 5. Show image

plt.imshow(newImage)
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

