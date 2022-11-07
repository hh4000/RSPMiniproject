import numpy as np
import matplotlib.pyplot as plt

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

forest = Tile("f.dat","forest")
ocean  = Tile("o.dat","ocean")
grass = Tile("g.dat","grasslands")
swamp = Tile("s.dat","Swamp")
mountain=Tile("m.dat","Mountain")
#forestC = Tile("fc.dat","forestC")
#oceanC = Tile("oc.dat","oceanC")
#grassC = Tile("gc.dat","grasslandsC")
#swampC = Tile("sc.dat","SwampC")
#mountainC=Tile("mc.dat","MountainC")
home=Tile("home.dat","Home")
#forestCrown = Tile("fc.dat","forest with crown")

arr = [forest,ocean,grass,swamp,mountain,home]#forestC,oceanC,grassC,swampC,mountainC,
names = []
x1 = np.linspace(0,179,1000)
x2 = np.linspace(0,255,1000)
x3 = np.linspace(0,75,500)
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