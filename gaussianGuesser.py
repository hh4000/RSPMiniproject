from imageSegmentation import segment
import numpy as np
import cv2 as cv
import dataComparison as dc

path = r"C:\Users\hansh\OneDrive - Aalborg Universitet\Programmer\3. semester\RSP\Miniproject\TrainingCropped\66.jpg"

inputImg = cv.imread(path,1)
tileArr = [dc.forest,dc.ocean,dc.grass,dc.swamp,dc.mountain,dc.wheat,dc.null]
def guessTiles(image):
    segmentedImage = segment(image)
    result = np.chararray((5,5))
    
    #This part no worke
    HPose = []
    highestlikelihood = -np.inf
    for y,imgrow in enumerate(segmentedImage):
        for x,img in enumerate(imgrow):
            likelihood = dc.home.calcLogLikelihood(img)
            if likelihood > highestlikelihood:
                highestlikelihood = likelihood
                HPose = [y,x]
    y,x=HPose
    result[y,x] = dc.home.name
    print((np.max(highestlikelihood)))
    for y,imgrow in enumerate(segmentedImage):
        for x,img in enumerate(imgrow):
            if result[y,x] != b"H":
                likelihoodArr = [tile.calcLogLikelihood(img) for tile in tileArr ]
                likelihoodArr[6] = likelihoodArr[6]-np.log(5)
                maxIndex = likelihoodArr.index(max(likelihoodArr))
                result[y,x] = tileArr[maxIndex].name
    return result

print(guessTiles(inputImg))
cv.imshow("original image", inputImg)
cv.waitKey(0)