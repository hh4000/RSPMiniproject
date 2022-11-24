import numpy as np
import cv2 as cv
from gaussianGuesser import guessTiles


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
    numCrowns = np.array([[2,1,0,0,0], #Placeholder until function is available
                          [0,0,0,0,0],
                          [0,1,0,0,1],
                          [0,1,0,1,0],
                          [0,0,1,0,1]])
    grassfire = np.array([[1,2,4,4,4], #Placeholder until function is available
                          [2,2,4,4,4],
                          [5,2,6,4,7],
                          [3,4,4,4,7],
                          [0,0,0,4,7]])
    tiles = np.zeros((len(np.unique(grassfire))))#Array of connected pieces
    crowns= np.zeros((len(np.unique(grassfire))))#Array of crowns of respective connected pieces of tiles array
    #nested for loop iterates through board
    for y in range(5):
        for x in range(5):
            id = grassfire[y,x] #tile grassfire id
            tiles[id]+=1 #adds to tilecount of found id
            crowns[id]+=numCrowns[y,x] #checks num of crowns and adds to crown array
    return int(tiles@crowns) #returns integer value of dotproduct of tiles and crowns vector