import cv2
import numpy as np


img = cv2.imread(r'C:\Users\silas\Desktop\Robotics 3. semester\Image processing miniproject\RSPMiniproject\Training set\1.jpg',1)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow('hsv',hsv)


#A function that takes the hue, and saturation(%) and value(%) (hsv) and detects the chosen colors
def tileDetection(hueLow,hueUp,satLow, satUp, valLow, valUp):
    #We turn the percentages into 8 bit intergers
    #The values are then inserted into two arrays for the lower and upper bound
    lowerBound = np.array([int(hueLow/1.411),int(satLow*2.55),int(valLow*2.55)])
    upperBound = np.array([int(hueUp/1.411), int(satUp*2.55), int(valUp*2.55)])
    #The bounds are then put into an array that can be returned
    bounds = np.array([lowerBound,upperBound])
    #The mask uses the lower and upper bounds to tell wether to save the pixel or not
    mask = cv2.inRange(hsv,bounds[0], bounds[1])
    #This uses the mask to check for the colors then only shows the colors that fit within the threshold
    result = cv2.bitwise_and(img, img, mask=mask)
    return result


#I have found values for the grasslands and have inserted them into the funtion
grassland = tileDetection(50,100,70,100,55,65) 

def closing(tile,kernelSize):
    #Turning the image binary so that "closing" can be performed
    result_gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
    (thresh, blackAndWhiteImage) = cv2.threshold(result_gray, 10, 255, cv2.THRESH_BINARY)
    #You determine the maximum size of the kernel you want to use
    closing = 0
    #The loop goes through many closings of increasing sizes
    #This should remove the small mess in the begining while closing the big holes in the end
    for x in range(kernelSize):
        #The size of the kernel changes along with how far the loop is
        kernel = np.ones([x+1,x+1],np.uint8)
        #The closing is then performed
        closing = cv2.morphologyEx(blackAndWhiteImage, cv2.MORPH_CLOSE, kernel)
    return closing

closedGrassland = closing(grassland,42)

#Showing all the images
cv2.imshow('result',grassland)
cv2.imshow('closing', closedGrassland)
cv2.waitKey()

#Hvis vi har det i HSV kunne man så måske bruge en template af en hsv farvet krone fordi den er rød med gul-ish i midten
#En I hver rotation selvfølgelig
