import cv2 as cv
import numpy as np
from imageSegmentation import segment
import os
import shutil

## Below makes a list including the destination to all images of the cropped training set
trainingImages = []
for root, dirs, files in os.walk("TrainingCropped"):
    for file in files:
        if file.endswith(".jpg"):
            trainingImages.append(os.path.join(root,file))


def testCorrect():
    """Image to confirm if any terminal output is correct. 
    Inputting 'Y' into the terminal continues the program
    Inputting 'N' into the program quits the program
    Any other input restarts the function
    """
    print("Is this correct? \"Y\" or \"N\"")
    answer = input()
    if answer == "Y":
        pass
    elif answer == "N":
        quit()
    else:
        print("Invalid answer")
        testCorrect()

typeDict = {
    'g':"g.dat",
    'f':"f.dat",
    'o':"o.dat",
    's':"s.dat",
    'w':"w.dat",
    'm':"m.dat",
    'h':"home.dat",
    'n':"null.dat"
}

for i, path in enumerate(trainingImages):
    
    print("\nTraining on image ", i+1, " of ", len(trainingImages))
    img = cv.imread(path,1)

    print("Please give answer key of \"",path,"\"\n")
    
    imgWithAnswers = img
    answerkey = []
    for y in range(5):
        answerkey.append([])
        for x in range(5):
            print("Current key:\n",[print(row) for row in answerkey],"\nPlease input key of tile (",y,",",x,")\n")
            type = input()
            answerkey[y].append(type)
            imgWithAnswers = cv.putText(imgWithAnswers,type,(x*100+25,y*100+75),cv.FONT_ITALIC,2,(30,30,220),4)
            
    cv.imshow("Image with answer key", imgWithAnswers)
    cv.waitKey(0)
    answerkey = np.array(answerkey)        
    [print(row) for row in answerkey]
    testCorrect()
    hsvImg = cv.cvtColor(img,cv.COLOR_BGR2HSV)
    segmentedHSV = np.array(segment(hsvImg))
    for y in range(5):
        for x in range(5):
            subImg = segmentedHSV[y,x]
            meanHue = np.mean(subImg[:,:,0])
            meanSat = np.mean(subImg[:,:,1])
            meanVal = np.mean(subImg[:,:,2])
            hueSTD = np.std(subImg[:,:,0])
            satSTD = np.std(subImg[:,:,1])
            valSTD = np.std(subImg[:,:,2])
            dataFile = "data/" + typeDict[answerkey[y,x]]
            f = open(dataFile,"a")
            arr = np.array([[meanHue,hueSTD,meanSat,satSTD,meanVal,valSTD]])
            np.savetxt(f,arr,"%f")
            f.close()
    shutil.move(path,"trainedImages")
#img = cv.imread(path,1)
#answerKey = []
#for y in range(5):
#    answerKey.append([])
#    for x in range(5):
#        print("type of tile: " +str(x+1)+","+str(y+1))
#        type = input()
#        answerKey[y].append(type)
#        img = cv.putText(img,type,(x*100+25,y*100+75),cv.FONT_ITALIC,2,(30,30,220),4)
#[print(" ".join(row)) for row in answerKey]
#cv.imshow("Image with answerkey",img)
#cv.waitKey()
#testCorrect()
#

#    
#}
#img = cv.imread(path, 1)
#hsvImg = cv.cvtColor(img,cv.COLOR_BGR2HSV)
#segmentedImg = np.array(segment(hsvImg))
#for y in range(5):
#    for x in range(5):

#        file = typeDict[answerKey[y,x]]
#        f = open(file,"a")
#        arr = np.array([[meanHue,hueSTD,meanSat,satSTD,meanVal,valSTD]])
#        np.savetxt(f,arr,"%f")
#        f.close()
#        #print("Segment " + str(x+1) + ", " +str(y+1))
#        #print("mean hue: " + str(meanHue))
#        #print("mean saturation: " + str(meanSat))
#        #print("mean value: " + str(meanValue))
#        #print("Hue Standard spread: " + str(hueSTD))
#        #print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
#        #cv.imshow("picture",segment(img)[y][x])
#        #cv.waitKey(0)
#        
#
##f  = open("test.dat","a")
##arr = np.array([[1,2,3],[4,5,6],[7,8,9]])
##np.savetxt(f,arr,"%d",header="somethin 1, somethin 2, my asshole")
##
##new = np.loadtxt("test.dat")[:,1]
##print(new)
##f.close