import cv2 as cv
import numpy as np

#silas path
#path = r'C:\Users\silas\Desktop\Robotics 3. semester\Image processing miniproject\RSPMiniproject\Training set\2.jpg'
#hans path
path = r'C:\Users\hansh\OneDrive - Aalborg Universitet\Programmer\3. semester\RSP\Miniproject\Training set\2.jpg'

img = cv.imread(path,1)

hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
cv.imshow('hsv',hsv)


#A function that takes the hue, and saturation(%) and value(%) (hsv) and detects the chosen colors
def tileDetection(hueLow,hueUp,satLow, satUp, valLow, valUp):
    #We turn the percentages into 8 bit intergers
    #The values are then inserted into two arrays for the lower and upper bound
    lowerBound = np.array([int(hueLow/2),int(satLow*2.55),int(valLow*2.55)])
    upperBound = np.array([int(hueUp/2), int(satUp*2.55), int(valUp*2.55)])
    #The bounds are then put into an array that can be returned
    bounds = np.array([lowerBound,upperBound])
    #The mask uses the lower and upper bounds to tell wether to save the pixel or not
    mask = cv.inRange(hsv,bounds[0], bounds[1])
    #This uses the mask to check for the colors then only shows the colors that fit within the threshold
    result = cv.bitwise_and(img, img, mask=mask)
    medianBlur = cv.medianBlur(result,5)
    return medianBlur

#A function that "closes" the tile type 
def closing(tile,kernelSize):
    #Turning the image binary so that "closing" can be performed
    result_gray = cv.cvtColor(tile, cv.COLOR_BGR2GRAY)
    (thresh, blackAndWhiteImage) = cv.threshold(result_gray, 10, 255, cv.THRESH_BINARY)
    #You determine the maximum size of the kernel you want to use
    closing = 0
    #The loop goes through many closings of increasing sizes
    #This should remove the small mess in the begining while closing the big holes in the end
    for x in range(kernelSize):
        #The size of the kernel changes along with how far the loop is
        kernel = np.ones([x+1,x+1],np.uint8)
        #The closing is then performed
        closing = cv.morphologyEx(blackAndWhiteImage, cv.MORPH_CLOSE, kernel)
    return closing

#I have found values for each tile type and have inserted them into the funtions
grassland = tileDetection(70,95,70,90,50,70) 
closedGrassland = closing(grassland,42)

ocean = tileDetection(190,215,0,100,0,100)
closedOcean = closing(ocean,42)

forest = tileDetection(90,110,30,70,10,50)
closedForest = closing(forest,45)

desert = tileDetection(45,55,85,100,70,85)
closedDesert = closing(desert,42)

swamp = tileDetection(35,55,30,60,35,50)
closedSwamp = closing(swamp,42)

mountain = tileDetection(0,179,0,20,0,20)
closedmountain = closing(mountain,61)

#Showing all the images
#cv.imshow('grassland',grassland)
#cv.imshow('closedGrassland', closedGrassland)
#cv.imshow('ocean',ocean)
#cv.imshow('closedOcean', closedOcean)
#cv.imshow('forest',forest)
#cv.imshow('closedForest',closedForest)
#cv.imshow('desert',desert)
#cv.imshow('closedDesert',closedDesert)
#cv.imshow('swamp',swamp)
#cv.imshow('closedSwamp',closedSwamp)
#cv.imshow('mountain',mountain)
#cv.imshow('closedmountain',closedmountain)



cv.waitKey()

#Hvis vi har det i HSV kunne man så måske bruge en template af en hsv farvet krone fordi den er rød med gul-ish i midten
#En I hver rotation selvfølgelig
