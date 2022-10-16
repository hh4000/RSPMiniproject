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
    tileTypes = [closedGrassland,closedOcean,closedForest,closedDesert,closedSwamp,closedMountain]
    tileChar = ['g','o','f','d','s','m']
    tiles = np.chararray((5,5))
    tiles[:] = ""
    
    for n, tileType in enumerate(tileTypes):
        segmentedBoard = segment(tileType)
        for y, imgRow in enumerate(segmentedBoard):
            for x, img in enumerate(imgRow):
                if np.median(img) == 255:
                    if tiles[y,x] != "":
                        tiles[y,x] = "E"
                    else:
                        tiles[y,x] = tileChar[n]
    for tileRow in tiles:
        for tile in tileRow:
            if tile == "":
                tile = "Ø"
    return tiles

print(identifyTiles())

cv.imshow("closedGrassland",closedGrassland)
cv.imshow("closedOcean",closedOcean)
cv.imshow("closedForest",closedForest)
cv.imshow("closedDesert",closedDesert)
cv.imshow("closedSwamp",closedSwamp)
cv.imshow("closedMountain",closedMountain)
cv.waitKey(0)