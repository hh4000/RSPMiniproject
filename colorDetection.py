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
mask = cv2.inRange(hsv,grassland[0], grassland[1])
print(mask)
result = cv2.bitwise_and(img, img, mask=mask)
cv2.imshow('result',result)
cv2.waitKey()


