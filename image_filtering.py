'''Credit : Bimal kumar Sah '''


import cv2
import numpy as np

def min(a,b):
    if a>b:
        return b
    else:
        return a

def max(a,b):
    if a>b:
        return a
    else:
        return b

# Min filter to remove salt noise
def min_filter(res,i,j):
    if(i>0 and j>0 and i<res.shape[0] and j<res.shape[1]):
        m = 255

        for x in range(3):
            for y in range(3):
                m = min(m,res[i-1+x][j-1+y])
        
        return m
    else:
        pass


# max filter to remove pepper noise
def max_filter(res,i,j):
    if(i>0 and j>0 and i<res.shape[0] and j<res.shape[1]):
        m = 255

        for x in range(3):
            for y in range(3):
                m = max(m,res[i-1+x][j-1+y])
        
        return m
    else:
        pass


# Average filter to remove both salt and pepper noise
def avg_filter(res,i,j):
    if(i>0 and j>0 and i<res.shape[0] and j<res.shape[1]):
        m = 0

        for x in range(3):
            for y in range(3):
                m = m + res[i-1+x][j-1+y]
        
        return m/9
    else:
        pass


img = cv2.imread('noisy_image.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print img.dtype
print img.shape

# Creating a 2D array of img.size + 2
res = np.zeros((img.shape[0]+2,img.shape[1]+2), dtype=img.dtype)
print res.shape

''' Replacing the inner element 
    by the pixel values of original image '''

for i in range(1,301):
    for j in range(1,301):
        res[i][j] = img[i-1][j-1]

# Filtering the image
img = np.zeros((img.shape), dtype=img.dtype)
print img.shape

for i in range(1,301):
    for j in range(1,301):
        img[i-1][j-1] = avg_filter(res,i,j)

cv2.imshow('image',img)
cv2.imshow('zero',res)
cv2.waitKey(0)
cv2.destroyAllWindows()