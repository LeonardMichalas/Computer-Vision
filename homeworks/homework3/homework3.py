import cv2 as cv2
import numpy as np

sample = cv2.imread('doku.png', 0)

def pxIsInImgRange(image, x, y):
    h,w = image.shape[:2]
    if (0<=x) and (x < w): #checks wether pixel is bigger 0 and smaller then right image border
            if (0<=y) and (y < h): #checks wether pixel is bigger 0 and smaller then bottom image border
                return True
    return False


def smooth(image, kernel, filter):
    img_copy = np.zeros(image.shape)
    h,w = image.shape[:2]
    radius = (kernel-1)/2
    

    for x in range (0, w):
        for y in range (0, h):
                    
            px = 0
            
            for vx2 in range (-radius, radius+1): #checks the values around the center
                for vy2 in range (-radius, radius+1): #checks the values around the center
                    x2 = x + vx2 #sets the spectated position on the shifted value 
                    y2 = y + vy2
                    if pxIsInImgRange(image,x2,y2): #checks if the spectated position is within the imager borders
                        px = px + ((image[y2][x2]*filter[vy2][vx2])/float(kernel*kernel))
                    else:
                        px = px + 0 #if its outside it adds nothing

            #After the two loops we put the summed average of the patch into the center of the batch.
            img_copy[y][x] = px  

    return img_copy

def conv_transform(image):

    image_copy = image.copy()

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image_copy[i][j] = image[image.shape[0]-i-1][image.shape[1]-j-1]

    return image_copy

#Applying a kernel on the given image
def conv(image, kernel):
    kernel = conv_transform(kernel)
    
    image_h, image_w = image.shape[:2]

    kernel_h, kernel_w = kernel.shape[:2]

    h = kernel_h//2
    w = kernel_w//2

    image_conv = np.zeros(image.shape)

    #looping through the image
    for y in range(h, image_h-h):
        for x in range(w, image_w-w):
            
            sum = 0

            #looping through the the patches of the images (size of the given kernel)
            for y2 in range(kernel_h):
                for x2 in range(kernel_w):
                    sum = (sum + image[y-h+y2][x-w+x2]*kernel[y2][x2])

            image_conv[y][x] = sum

    return image_conv

def norm(img1, img2):

    threshold = 90

    img_copy = np.zeros(img1.shape) #new empty image
    image_h, image_w = img1.shape[:2]

    for y in range(0, image_h):
        for x in range(0, image_w):

            #Some values of the new images are negative, if they are negative we make them positive again here
            if (img1[y][x] < 0) : img1[y][x] = img1[y][x] * (-1)
            if (img2[y][x] < 0) : img2[y][x] = img2[y][x] * (-1)

            #Here we add the values of the two images    
            q = img1[y][x] + img2[y][x]

            if (q>threshold):
                img_copy[y][x] = 255 #by doing this we get an binary image
            else:
                img_copy[y][x] = 0

    return img_copy  

filter = np.zeros((3, 3), dtype = 'int')

#Fills smoothing Filter
filterValues = np.array([1,1,1])

for x in range (0, filterValues.size):
    for y in range (0, filterValues.size):

        filter[y][x] = filterValues[x] 
           
#here we are smoothing the image first
sample = smooth(sample, 3, filter)

#this is our veritcal filter     
edgerVer = np.zeros(shape = (3, 3))

edgerVer[0][0] = -1 
edgerVer[1][0] = -2
edgerVer[2][0] = -1
edgerVer[0][1] = 0
edgerVer[1][1] = 0
edgerVer[2][1] = 0
edgerVer[0][2] = 1
edgerVer[1][2] = 2
edgerVer[2][2] = 1 

gy = conv(sample, edgerVer)
cv2.imshow('Y_Sobel', gy)

#this is our horizontal filter  
edgerHor = np.zeros(shape = (3, 3))

edgerHor[0][0] = -1 
edgerHor[1][0] = 0
edgerHor[2][0] = 1 
edgerHor[0][1] = -2
edgerHor[1][1] = 0
edgerHor[2][1] = 2
edgerHor[0][2] = -1 
edgerHor[1][2] = 0
edgerHor[2][2] = 1 

gx = conv(sample, edgerHor)
cv2.imshow('X_Sobel', gx)

g_sobel = norm(gy, gx)
cv2.imshow('Sobel_Edge_Detection', g_sobel)

print('done...')
cv2.waitKey(0)
cv2.destroyAllWindows()













