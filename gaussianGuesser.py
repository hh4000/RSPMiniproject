from imageSegmentation import segment
import numpy as np
import cv2 as cv
import dataComparison as dc

path = r"C:\Users\hansh\OneDrive - Aalborg Universitet\Programmer\3. semester\RSP\Miniproject\Training set\17.jpg"

inputImg = cv.imread(path,1)
tileArr = [dc.forest,dc.ocean,dc.grass,dc.swamp,dc.mountain,dc.wheat,dc.home,dc.forestC,dc.oceanC,dc.grassC,dc.swampC,dc.mountainC,dc.wheatC]
def guessTiles(image):
    segmentedImage = segment(image)
    result = np.chararray((5,5))
    for y,imgrow in enumerate(segmentedImage):
        for x,img in enumerate(imgrow):
            likelihoodArr = [tile.calcLogLikelihood(img) for tile in tileArr ]
            maxIndex = likelihoodArr.index(max(likelihoodArr))
            result[y,x] = tileArr[maxIndex].name
    return result

print(guessTiles(inputImg))
cv.imshow("original image", inputImg)
cv.waitKey(0)