import cv2
import numpy as np


img = cv2.imread(r'C:\Users\silas\Desktop\Robotics 3. semester\Image processing miniproject\RSPMiniproject\Training set\1.jpg',1)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow('hsv',hsv)


#A function that takes the hue, and saturation(%) and value(%) (hsv)
def lowUpBound(hueLow,hueUp,satLow, satUp, valLow, valUp):
    #We take the percentages
    satUpTemp = satUp
    satLowTemp = satLow
    valueLowTemp = valLow
    valueUpTemp = valUp
    #Turn them into 8 bit intergers
    lowerSaturation = int(satLowTemp*2.55)
    upperSaturation = int(satUpTemp*2.55)
    lowerValue = int(valueLowTemp*2.55)
    upperValue = int(valueUpTemp*2.55)
    #The values are then inserted into two arrays for the lower and upper bound
    lowerBound = np.array([hueLow,lowerSaturation,lowerValue])
    upperBound = np.array([hueUp, upperSaturation, upperValue])
    #The bounds are then put into an array that can be returned
    bounds = np.array([lowerBound,upperBound])
    return bounds


#I have found values for the grasslands and have created lower and upper bounds 
grassland = lowUpBound(40,100,50,100,50,100)
#The mask uses the lower and upper bounds to tell wether to save the pixel or not
mask = cv2.inRange(hsv,grassland[0], grassland[1])
#This uses the mask to check for the colors then only shows the colors that fit within the threshold
result = cv2.bitwise_and(img, img, mask=mask)

#Turning the image binary so that "closing" can be performed
result_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
(thresh, blackAndWhiteImage) = cv2.threshold(result_gray, 10, 255, cv2.THRESH_BINARY)

#You determine the maximum size of the kernel you want to use
kernelSize = 42
#The loop goes through many closings of increasing sizes
#This should remove the small mess in the begining while closing the big holes in the end
for x in range(kernelSize):
    global closing
    #The size of the kernel changes along with how far the loop is
    kernel = np.ones([x+1,x+1],np.uint8)
    #The closing is then performed
    closing = cv2.morphologyEx(blackAndWhiteImage, cv2.MORPH_CLOSE, kernel)

#Showing all the images
cv2.imshow('binary',blackAndWhiteImage)
cv2.imshow('result',result)
cv2.imshow('closing', closing)
cv2.waitKey()


