import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

class Tile:
    def __init__(self,fileName,type):
        self.data = np.loadtxt(fileName)
        self.hueMeans = self.data[:,0]
        self.hueSTDs  = self.data[:,1]
        self.satMeans = self.data[:,2]
        self.satSTDs  = self.data[:,3]
        self.valMeans = self.data[:,4]
        self.valSTDs  = self.data[:,5]
        self.name = type
        self.numPoints=len(self.data)
    def normDist(self,x,mean,std):
        return 1/(std*np.sqrt(2*np.pi))*np.e**(-1/2*((x-mean)/std)**2)
    
    def calibrateForOccurance(self,totalPoints):
        self.occurance = self.numPoints/totalPoints
        #print("Occurance set to: ", self.occurance)
    def calcLogLikelihood(self,img):
        img = cv.cvtColor(img,cv.COLOR_BGR2HSV)
        hueMean = np.mean(img[:,:,0])
        hueSTD  = np.std(img[:,:,0])
        satMean = np.mean(img[:,:,1])
        satSTD  = np.std(img[:,:,1])
        valMean = np.mean(img[:,:,2])
        valSTD  = np.std(img[:,:,2])
        if self.normDist(hueMean,np.mean(self.hueMeans),np.std(self.hueMeans)) == 0 or self.normDist(hueSTD,np.mean(self.hueSTDs),np.std(self.hueSTDs)) == 0:
            return -np.inf
        hueLikelihood = np.log(self.normDist(hueMean,np.mean(self.hueMeans),np.std(self.hueMeans)))+np.log(self.normDist(hueSTD,np.mean(self.hueSTDs),np.std(self.hueSTDs)))
        if self.normDist(satMean,np.mean(self.satMeans),np.std(self.satMeans)) == 0 or self.normDist(satSTD,np.mean(self.satSTDs),np.std(self.satSTDs)) == 0:
            return -np.inf
        satLikelihood = np.log(self.normDist(satMean,np.mean(self.satMeans),np.std(self.satMeans)))+np.log(self.normDist(satSTD,np.mean(self.satSTDs),np.std(self.satSTDs)))
        if self.normDist(valMean,np.mean(self.valMeans),np.std(self.valMeans)) == 0 or self.normDist(valSTD,np.mean(self.valSTDs),np.std(self.valSTDs)) == 0:
            return -np.inf
        valLikelihood = np.log(self.normDist(valMean,np.mean(self.valMeans),np.std(self.valMeans)))+np.log(self.normDist(valSTD,np.mean(self.valSTDs),np.std(self.valSTDs)))
        return hueLikelihood + satLikelihood + valLikelihood
forest = Tile("data/f.dat","f")
ocean  = Tile("data/o.dat","o")
grass  = Tile("data/g.dat","g")
swamp  = Tile("data/s.dat","s")
mountain=Tile("data/m.dat","m")
wheat  = Tile("data/w.dat","w")
home   = Tile("data/home.dat","H")
null   = Tile("data/null.dat","n")

totalNumPoints = forest.numPoints+ocean.numPoints+grass.numPoints+swamp.numPoints+mountain.numPoints+wheat.numPoints+home.numPoints+null.numPoints
#print(totalNumPoints)

forest.calibrateForOccurance(totalNumPoints)
ocean.calibrateForOccurance(totalNumPoints)
grass.calibrateForOccurance(totalNumPoints)
swamp.calibrateForOccurance(totalNumPoints)
mountain.calibrateForOccurance(totalNumPoints)
wheat.calibrateForOccurance(totalNumPoints)
home.calibrateForOccurance(totalNumPoints)
null.calibrateForOccurance(totalNumPoints)
#for a in [forest,ocean,grass,swamp,mountain,wheat,home,null]:
#    print(a.name,": ",a.occurance)

#forestCrown = Tile("fc.dat","forest with crown")
def compareData():
    arr = [forest,ocean,grass,swamp,mountain,wheat,home,null]
    names = []
    x1 = np.linspace(0,179,1000)
    x2 = np.linspace(0,255,1000)
    x3 = np.linspace(0,100,500)
    fig,ax = plt.subplots(2,3)
    def normDist(x,std,mean):
        return 1/(std*np.sqrt(2*np.pi))*np.e**(-1/2*((x-mean)/std)**2)
    for type in arr:
        ax[0,0].plot(x1,normDist(x1,np.std(type.hueMeans),np.mean(type.hueMeans)))
        ax[0,1].plot(x2,normDist(x2,np.std(type.satMeans),np.mean(type.satMeans)))
        ax[0,2].plot(x2,normDist(x2,np.std(type.valMeans),np.mean(type.valMeans)))
        
        ax[1,0].plot(x3,normDist(x3,np.std(type.hueSTDs),np.mean(type.hueSTDs)))
        ax[1,1].plot(x3,normDist(x3,np.std(type.satSTDs),np.mean(type.satSTDs)))
        ax[1,2].plot(x3,normDist(x3,np.std(type.valSTDs),np.mean(type.valSTDs)))
        names.append(type.name)
    ax[0,0].set_title("mean hue")
    ax[0,1].set_title("mean saturation")
    ax[0,2].set_title("mean value")
    ax[1,0].set_title("Hue std")
    ax[1,1].set_title("Saturation std")
    ax[1,2].set_title("Value std")
    ax[0,0].legend(names)
    plt.show()
#compareData()