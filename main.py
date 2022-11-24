import numpy as np
import cv2 as cv
from gaussianGuesser import guessTiles
from Grassfire import grassfire
from Crown_detection import ToHans as crownDetection

image = cv.imread("testCropped/7.jpg",1)
def calculateScore(img):
    """Takes 500x500 pixel cropped perspectivecorrected image of 
    king domino board and calculates point value

    Args:
        img (mat): 500x500 pixel cropped and perspective corrected BGR Image

    Returns:
        int: point value of board given
    """
    guessedTiles = guessTiles(img)
    numCrowns = crownDetection(img)
    objects = grassfire(guessedTiles)
    tiles = np.zeros((len(np.unique(objects))))#Array of connected pieces
    crowns= np.zeros((len(np.unique(objects))))#Array of crowns of respective connected pieces of tiles array
    #nested for loop iterates through board
    for y in range(5):
        for x in range(5):
            id = int(objects[y,x]) #tile grassfire id
            tiles[id]+=1 #adds to tilecount of found id
            crowns[id]+=numCrowns[y,x] #checks num of crowns and adds to crown array
    return int(tiles@crowns) #returns integer value of dotproduct of tiles and crowns vector
print(calculateScore(image))