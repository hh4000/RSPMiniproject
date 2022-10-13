import cv2
import numpy as np


img = cv2.imread(r'C:\Users\silas\Desktop\Robotics 3. semester\Image processing miniproject\RSPMiniproject\Training set\1.jpg',1)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow('hsv',hsv)


#A function that takes the hue, and saturation(%) and value(%) (hsv) and detects the chosen colors
def tileDetection(hueLow,hueUp,satLow, satUp, valLow, valUp):
    #We turn the percentages into 8 bit intergers
    #The values are then inserted into two arrays for the lower and upper bound
    lowerBound = np.array([int(hueLow/2),int(satLow*2.55),int(valLow*2.55)])
    upperBound = np.array([int(hueUp/2), int(satUp*2.55), int(valUp*2.55)])
    #The bounds are then put into an array that can be returned
    bounds = np.array([lowerBound,upperBound])
    #The mask uses the lower and upper bounds to tell wether to save the pixel or not
    mask = cv2.inRange(hsv,bounds[0], bounds[1])
    #This uses the mask to check for the colors then only shows the colors that fit within the threshold
    result = cv2.bitwise_and(img, img, mask=mask)
    medianBlur = cv2.medianBlur(result,5)
    return medianBlur

#A function that "closes" the tile type 
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

#I have found values for each tile type and have inserted them into the funtions
grassland = tileDetection(70,95,70,90,50,70) 
closedGrassland = closing(grassland,42)

ocean = tileDetection(190,215,0,100,0,100)
closedOcean = closing(ocean,42)

#It kinda works, but is a little not optimal
forest = tileDetection(90,110,30,70,10,50)
closedForest = closing(forest,45)

desert = tileDetection(45,55,85,100,70,85)
closedDesert = closing(desert,42)

swamp = tileDetection(35,55,30,60,35,50)
closedSwamp = closing(swamp,42)


#Showing all the images
#cv2.imshow('grassland',grassland)
#cv2.imshow('closedGrassland', closedGrassland)
#cv2.imshow('ocean',ocean)
#cv2.imshow('closedOcean', closedOcean)
#cv2.imshow('forest',forest)
#cv2.imshow('closedForest',closedForest)
#cv2.imshow('desert',desert)
#cv2.imshow('closedDesert',closedDesert)
#cv2.imshow('swamp',swamp)
#cv2.imshow('closedSwamp',closedSwamp)




cv2.waitKey()

#Hvis vi har det i HSV kunne man så måske bruge en template af en hsv farvet krone fordi den er rød med gul-ish i midten
#En I hver rotation selvfølgelig
