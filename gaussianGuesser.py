from imageSegmentation import segment
import numpy as np
import cv2 as cv
import dataComparison as dc

###image used for testing
#path = r"C:\Users\hansh\OneDrive - Aalborg Universitet\Programmer\3. semester\RSP\Miniproject\TrainingCropped\5.jpg"
#inputImg = cv.imread(path,1)


def guessTiles(image):
    """Takes image of 5x5 king domino board and assign tile types

    Args:
        image (mat): 500x500 pixel img of king domino board
        must be perspective corrected and cropped

    Returns:
        numpy array of chars: array of chars describing the type and position of tiles - 
        n is null
        H is home
        f is forest
        g is grass
        m is mountain
        o is ocean
        s is swamp
        w is field (of wheat)
    """
    segmentedImage = segment(image) #segments image into 25 tiles
    result = np.chararray((5,5)) # initializes result array
    tileArr = [dc.forest,dc.ocean,dc.grass,dc.swamp,dc.mountain,dc.wheat,dc.null] #Array of non-home tiles
    
    
    #The following code paragraph guesses the position of 
    # the home tile, while considering there may only be 1
    HPoseArr = []
    HlikelihoodArr = []
    homePose = []
    #Below nested array iterates through all 25 tiles to figure out which may be the home tile
    for y,imgrow in enumerate(segmentedImage):
        for x,img in enumerate(imgrow):
            HLogLikelihood = dc.home.calcLogLikelihood(img)
            likelihoodArr = [tile.calcLogLikelihood(img)+np.log(tile.occurence) for tile in tileArr]
            #if statement that find the tile with the largest likelihood 
            # while also excluding tiles with a higher likelihood of being a different tile
            if max(likelihoodArr) < HLogLikelihood:
                HPoseArr.append([y,x])
                HlikelihoodArr.append(HLogLikelihood)
    # If statement in the case where no home tile is found
    if len(HPoseArr)>0:
        maxIndex = HlikelihoodArr.index(max(HlikelihoodArr))                
        homePose=HPoseArr[maxIndex]
        result[homePose[0],homePose[1]] = dc.home.name
    # currently the code does not necessarily find a home tile, 
    # if it believes all tiles are more likely to be something else
    
    # Following paragraph iterates through the image and identifies the most likely tile type
    for y,imgrow in enumerate(segmentedImage):
        for x,img in enumerate(imgrow):
            if [y,x] != homePose: # Does not consider the tile designated as home tile
                likelihoodArr = [tile.calcLogLikelihood(img)+np.log(tile.occurence) for tile in tileArr]
                maxIndex = likelihoodArr.index(max(likelihoodArr))
                result[y,x] = tileArr[maxIndex].name

    # returns the char array decoded to only include the character 
    # (i.e. saying 'a' instead of b'a' (as an example))
    return result.decode() 