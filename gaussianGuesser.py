from imageSegmentation import segment
import numpy as np
import cv2 as cv
import dataComparison as dc

path = r"C:\Users\hansh\OneDrive - Aalborg Universitet\Programmer\3. semester\RSP\Miniproject\TrainingCropped\5.jpg"

inputImg = cv.imread(path,1)

def guessTiles(image):
    segmentedImage = segment(image)
    result = np.chararray((5,5))
    tileArr = [dc.forest,dc.ocean,dc.grass,dc.swamp,dc.mountain,dc.wheat,dc.null]
    #This part no worke
    
    HPoseArr = []
    HlikelihoodArr = []
    homePose = []
    for y,imgrow in enumerate(segmentedImage):
        for x,img in enumerate(imgrow):
            HLogLikelihood = dc.home.calcLogLikelihood(img)
            likelihoodArr = [tile.calcLogLikelihood(img)+np.log(tile.occurence) for tile in tileArr]
            if max(likelihoodArr) < HLogLikelihood:
                HPoseArr.append([y,x])
                HlikelihoodArr.append(HLogLikelihood)
    if len(HPoseArr)>0:
        maxIndex = HlikelihoodArr.index(max(HlikelihoodArr))                
        homePose=HPoseArr[maxIndex]
        result[homePose[0],homePose[1]] = dc.home.name
    #print((np.max(highestlikelihood)))
    for y,imgrow in enumerate(segmentedImage):
        for x,img in enumerate(imgrow):
            if [y,x] != homePose :
                likelihoodArr = [tile.calcLogLikelihood(img)+np.log(tile.occurence) for tile in tileArr]
                maxIndex = likelihoodArr.index(max(likelihoodArr))

                result[y,x] = tileArr[maxIndex].name
    return result.decode()

#result = guessTiles(inputImg)
#print(result)
##cv.imshow("original image", inputImg)
#for y in range(5):
#    for x in range(5):
#        inputImg = cv.putText(inputImg,result[y,x],(x*100+25,y*100+75),cv.FONT_ITALIC,2,(30,30,220),4)
#cv.imshow("Image with answers", inputImg)
#cv.waitKey(0)

