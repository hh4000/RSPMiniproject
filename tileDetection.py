import cv2 as cv
import numpy as np

from imageSegmentation import segment
#import commands and variables from colorDetection
from colorDetection import closedGrassland
from colorDetection import closedOcean
from colorDetection import closedForest
from colorDetection import closedDesert
from colorDetection import closedSwamp
from colorDetection import closedMountain

def identifyTiles():#must be changed to take argument (for varying input image)
    """From a cropped, perspective corrected image of a board, identifies the types of tiles present on the board.
    

    Returns:
        numpy.chararray: char array with identified tile types. Grasslands = 'g', Ocean = 'o', Forest = 'f', Desert = 'd', Swamp = 's', Mountain = 'm', Error = 'E', Unidentified = 'Ø'
    """
    #array of closed colordetected image
    tileTypes = [closedGrassland,closedOcean,closedForest,closedDesert,closedSwamp,closedMountain]
    #array of characters used to identify the tile types - grasslands are 'g', oceans are 'o', etc.
    tileChar = ['g','o','f','d','s','m']
    
    #initializes output array with all inputs as ""
    tiles = np.chararray((5,5))
    tiles[:] = ""
    
    for n, tileType in enumerate(tileTypes):#checks for each tile type
        segmentedBoard = segment(tileType)#segments into 25 (5x5) pieces
        for y, imgRow in enumerate(segmentedBoard): #checks each row of board
            for x, img in enumerate(imgRow):#checks each tile of row
                
                if np.median(img) == 255: #checks if tyle is of current tile Type - pressumed if >50% of image is white that it is of current type
                    #checks if current tile has already has a identified tile type. if it has, type is changed to 'E' (error)
                    if tiles[y,x] != "":
                        tiles[y,x] = "E"
                    else:
                    #if no error type is set accordingly
                        tiles[y,x] = tileChar[n]
                        
    # checks if any tiles are unidentfied. In that case, type is set as 'Ø'
    for y, tileRow in enumerate(tiles):
        for x, tile in enumerate(tileRow):
            if tile == "":
                tiles[y,x] = "Ø"
    return tiles

print(identifyTiles())

cv.imshow("closedGrassland",closedGrassland)
cv.imshow("closedOcean",closedOcean)
cv.imshow("closedForest",closedForest)
cv.imshow("closedDesert",closedDesert)
cv.imshow("closedSwamp",closedSwamp)
cv.imshow("closedMountain",closedMountain)
cv.waitKey(0)