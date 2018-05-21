#Assignment 1: Plot a historgram with matplotlib

import matplotlib.pyplot as plt
import numpy as np
import cv2

name = 'flower_grey_image.jpg'

img = cv2.imread(name, cv2.IMREAD_GRAYSCALE) #import image
newImg = np.zeros((img.shape))

def get_histo_scope(img):

    imgPixelList = [] #array which later can save the pixel values of the image

    h = img.shape[0] #number of pixels in the hight
    w = img.shape[1] #number of piexels in the weight

    darkestValue = 256 #oposite so it can get darker while loop
    whitestValue = 0 #oposite so it can get lighter while loop

    for y in range(0, w):
        for x in range(0, h):       
            px = img[x][y] #reads the pixel which is a npndarray [][][]
            imgPixelList.append(px) #saves the pixel data of every pixel we loop so we can use it later to plot the histogram
            if darkestValue > px: #identifies the darkest pixel value
                darkestValue = px
            if whitestValue < px: #identifies the whitest pixel value
                whitestValue = px 
              
    return darkestValue, whitestValue, imgPixelList

def plot(imgPixelList, darkestValue, whitestValue, title):
    values = range(darkestValue, whitestValue, 1) #creates and array with all data from whitesValue to darkestValue
    bin_edges = values

    plt.hist(imgPixelList, bins=bin_edges, color='black')
    plt.xlabel('Color Values')
    plt.ylabel('Number of Poxels')
    plt.title(title)
    plt.show()  

    return     

def equalize(img):
    hist,bins = np.histogram(img.flatten(),256,[0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max()/ cdf.max()

    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
    img = cdf[img]

    plt.hist(img.flatten(),256,[0,256], color = 'r')
    plt.xlim([0,256])
    plt.legend(('cdf','histogram'), loc = 'upper left')
    plt.title('Equalized Histogram')
    plt.show()

    img = cv2.imread(name,0)
    equ = cv2.equalizeHist(img)
    res = np.hstack((img,equ)) #stacking images side-by-side
    cv2.imwrite('comparison.png',res)

    return

def stratch_contrast(img, darkestValue, whitestValue): 

    newImgPixelList = []

    h = img.shape[0] #number of pixels in the hight
    w = img.shape[1] #number of piexels in the weight

    darkestValueStratch = 256 #oposite so it can get darker while loop
    whitestValueStratch = 0 #oposite so it can get lighter while loop

    newImg[0][0] = 256*(img[0][0] - darkestValue)/(whitestValue-darkestValue)

    for y in range(0, w):
       for x in range(0, h):

            newImg[x][y] = 256*(img[x][y] - darkestValue)/(whitestValue-darkestValue)
            pxStratch = newImg[x][y]
            print(pxStratch)
            newImgPixelList.append(pxStratch)
            if darkestValueStratch > pxStratch: #identifies the darkest pixel value
                darkestValueStratch = pxStratch
            if whitestValueStratch < pxStratch: #identifies the whitest pixel value
                whitestValueStratch = pxStratch            
                
    return newImgPixelList, darkestValueStratch, whitestValueStratch

darkestValue, whitestValue, imgPixelList = get_histo_scope(img) #get scope and pixel values from the img data

plot(imgPixelList, darkestValue, whitestValue, 'Normal Histogram') #plot the collected pixel values

equalize(img) #Equalize, plot and comparison picture


#NOT WORKING AS IT SHOULD

#newImgPixelList, darkestValueStratch, whitestValueStratch = stratch_contrast(img, darkestValue, whitestValue)

#plot(newImgPixelList, int(darkestValueStratch), int(whitestValueStratch), 'Equalized Histogram')