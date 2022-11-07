import cv2 as cv
import numpy as np

#Hans path (comment out if not Hans)
path = r"C:\Users\hansh\OneDrive - Aalborg Universitet\Programmer\3. semester\RSP\Miniproject\Training set\1.jpg"

#Hans path (comment out if not Hans)
#path = ---- ADD PATH HERE ----

def segment(img):
    """segments image into 5x5 tiles based on height and with

    Args:
        img (mat): image (either 3D matrix or 2D matrix)

    Returns:
        array: a 5x5 array width an image in each index
    """
    #calculates the height and width of each segment when images is split into 25 equal areas (5x5)
    segmentWidth = int(img.shape[1]/5)
    segmentHeight= int(img.shape[0]/5)
    #initializes output array 
    output = []
    for segmentY in range(5):
        output.append([]) #added line of images
        for segmentX in range(5):
            output[segmentY].append( img[segmentY*segmentHeight:(segmentY+1)*segmentHeight,segmentX*segmentWidth:(segmentX+1)*segmentWidth]) # inputs image into line
    return output 
            
testImg = cv.imread(path, 1)

#cv.imshow("picture",testImg)
#cv.waitKey(0)