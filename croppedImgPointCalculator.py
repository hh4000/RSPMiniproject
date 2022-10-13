import cv2 as cv
import numpy as np

#Hans path (comment out if not Hans)
path = r"C:\Users\hansh\OneDrive - Aalborg Universitet\Programmer\3. semester\RSP\Miniproject\Training set\1.jpg"

#Hans path (comment out if not Hans)
#path = ---- ADD PATH HERE ----

def segment(img):
    segmentWidth = int(img.shape[1]/5)
    segmentHeight= int(img.shape[0]/5)
    output = []
    for segmentY in range(5):
        output.append([])
        for segmentX in range(5):
            output[segmentY].append( img[segmentY*segmentHeight:(segmentY+1)*segmentHeight,segmentX*segmentWidth:(segmentX+1)*segmentWidth])
    return output 
            
testImg = cv.imread(path, 1)

cv.imshow("picture",testImg)
cv.waitKey(0)