import cv2 as cv
import numpy as np
from imageSegmentation import segment
path = r"C:\Users\hansh\OneDrive - Aalborg Universitet\Programmer\3. semester\RSP\Miniproject\Training set\9.jpg"

def testCorrect():
    print("Is this correct? \"Y\" or \"N\"")
    answer = input()
    if answer == "Y":
        pass
    elif answer == "N":
        quit()
    else:
        print("Invalid answer")
        testCorrect()

print(path)
testCorrect()

answerKey = np.chararray((5,5))
for y in range(5):
    for x in range(5):
        print("type of tile: " +str(x+1)+","+str(y+1))
        type = input()
        answerKey[y,x] = type
print(answerKey)

testCorrect()

typeDict = {
    b'g':"g.dat",
    b'G':"gc.dat",
    b'f':"f.dat",
    b'F':"fc.dat",
    b'o':"o.dat",
    b'O':"oc.dat",
    b's':"s.dat",
    b'S':"sc.dat",
    b'w':"w.dat",
    b'W':"wc.dat",
    b'm':"m.dat",
    b'M':"mc.dat",
    b'H':"home.dat",
    b'n':"null.dat"
    
}
img = cv.imread(path, 1)
hsvImg = cv.cvtColor(img,cv.COLOR_BGR2HSV)
segmentedImg = np.array(segment(hsvImg))
for y in range(5):
    for x in range(5):
        subImg = segmentedImg[y,x]
        meanHue = np.mean(subImg[:,:,0])
        meanSat = np.mean(subImg[:,:,1])
        meanVal = np.mean(subImg[:,:,2])
        hueSTD = np.std(subImg[:,:,0])
        satSTD = np.std(subImg[:,:,1])
        valSTD = np.std(subImg[:,:,2])
        file = typeDict[answerKey[y,x]]
        f = open(file,"a")
        arr = np.array([[meanHue,hueSTD,meanSat,satSTD,meanVal,valSTD]])
        np.savetxt(f,arr,"%f")
        f.close()
        #print("Segment " + str(x+1) + ", " +str(y+1))
        #print("mean hue: " + str(meanHue))
        #print("mean saturation: " + str(meanSat))
        #print("mean value: " + str(meanValue))
        #print("Hue Standard spread: " + str(hueSTD))
        #print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        #cv.imshow("picture",segment(img)[y][x])
        #cv.waitKey(0)
        

#f  = open("test.dat","a")
#arr = np.array([[1,2,3],[4,5,6],[7,8,9]])
#np.savetxt(f,arr,"%d",header="somethin 1, somethin 2, my asshole")
#
#new = np.loadtxt("test.dat")[:,1]
#print(new)
#f.close