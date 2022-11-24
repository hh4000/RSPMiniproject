import numpy as np
import cv2 as cv
from gaussianGuesser import guessTiles
answers = np.array([
    [['s', 'w', 'f', 'f', 'f'], #7.jpg
     ['w', 'w', 'f', 'f', 'f'], 
     ['m', 'w', 'H', 'f', 'o'], 
     ['w', 'f', 'f', 'f', 'o'], 
     ['o', 'o', 'o', 'f', 'o']], 
    
    [['w', 'w', 'g', 'g', 'n'], #12.jpg
     ['f', 'o', 'o', 'o', 'n'], 
     ['g', 'g', 'g', 'f', 'o'], 
     ['g', 's', 'g', 'w', 'o'], 
     ['s', 's', 'H', 'w', 'o']], 
    
    [['w', 'm', 'm', 'm', 'n'], #14.jpg
     ['o', 'f', 's', 's', 'n'], 
     ['o', 'o', 'H', 's', 'w'], 
     ['o', 'o', 'f', 'w', 'w'], 
     ['o', 'o', 'f', 'f', 'f']], 
    
    
    [['w', 'g', 'f', 'g', 'o'], #19.jpg
     ['w', 'g', 'g', 'g', 'w'], 
     ['m', 's', 'H', 'f', 'w'], 
     ['m', 'm', 'w', 'f', 's'], 
     ['w', 'w', 'w', 'w', 'w']],
    
    [['g', 'f', 'w', 'w', 'o'], #22.jpg
     ['n', 'f', 'f', 'w', 'w'], 
     ['n', 'f', 'H', 'w', 'w'], 
     ['f', 'f', 'f', 'f', 'f'], 
     ['o', 'o', 'f', 'f', 'f']],
    
    [['m', 'm', 'w', 'w', 's'], #28.jpg
     ['s', 's', 'g', 'g', 'f'],
     ['s', 's', 's', 'g', 'H'],
     ['s', 's', 's', 'g', 'g'],
     ['w', 'w', 'w', 'w', 'w']],
    
    [['f', 'f', 'f', 'f', 'o'], #30.jpg
     ['f', 'w', 'f', 'f', 'o'],
     ['m', 'w', 'H', 'f', 'o'],
     ['w', 'w', 'f', 'f', 'o'],
     ['o', 'o', 'o', 'o', 'o']],
    
    [['m', 'm', 'w', 'w', 's'], #32.jpg
     ['s', 's', 'g', 'g', 'f'],
     ['s', 's', 's', 'g', 'H'],
     ['s', 's', 's', 'g', 'g'],
     ['w', 'w', 'w', 'w', 'w']],
    
    [['g', 'w', 'o', 'o', 'o'], #33.jpg
     ['w', 'w', 's', 'w', 'f'],
     ['o', 'w', 'H', 'f', 'f'],
     ['o', 'f', 'f', 'f', 'n'],
     ['o', 'f', 'w', 'w', 'n']],
    
    [['s', 'w', 'f', 'f', 'f'], #35.jpg
     ['s', 'f', 'f', 'o', 'o'],
     ['w', 'w', 'H', 'o', 'o'],
     ['w', 'w', 'w', 'o', 'o'],
     ['m', 'm', 'm', 'm', 's']],
    
    [['f', 'w', 's', 'w', 'g'], #45.jpg
     ['f', 'w', 's', 's', 's'],
     ['H', 'w', 'm', 's', 's'],
     ['f', 'w', 'w', 'o', 'o'],
     ['f', 'f', 'f', 'o', 'o']],
    
    [['w', 'o', 'o', 'f', 'f'], #46.jpg
     ['w', 'w', 'f', 'f', 'f'],
     ['w', 'o', 'f', 'f', 's'],
     ['w', 'o', 'o', 'f', 'm'],
     ['w', 'w', 'w', 'f', 'H']],

    [['g', 'f', 'g', 'g', 'o'], #57.jpg
     ['g', 'f', 'g', 'w', 'f'],
     ['g', 'g', 'g', 'w', 'H'],
     ['g', 's', 'g', 'w', 'w'],
     ['g', 's', 'g', 'o', 'o']],
    
    [['w', 's', 's', 'w', 'H'], #62.jpg
     ['m', 'm', 'w', 'w', 'f'],
     ['m', 'w', 'w', 'f', 'f'],
     ['o', 'w', 'w', 'w', 'f'],
     ['g', 'g', 'w', 'f', 'f']],

    [['f', 'f', 'f', 'f', 'f'], #68.jpg
     ['f', 'w', 'w', 'w', 'w'],
     ['f', 'w', 'w', 'w', 'w'],
     ['f', 'w', 'f', 'f', 'w'],
     ['f', 'f', 'f', 'f', 'H']]
])
imgList = ["7.jpg","12.jpg","14.jpg","19.jpg","22.jpg","28.jpg","30.jpg","32.jpg","33.jpg","35.jpg","45.jpg","46.jpg","57.jpg","62.jpg","68.jpg"]

dictionary = {
    "n":0,
    "s":1,
    "w":2,
    "f":3,
    "m":4,
    "o":5,
    "g":6,
    "H":7
}
resultArr = np.zeros((8,8))
for n,imgName in enumerate(imgList):
    img = cv.imread("testCropped/"+imgName,1)
    result = guessTiles(img)
    for y in range(5):
        for x in range(5):
            i = dictionary[answers[n,y,x]] #The true result
            j = dictionary[result[y,x]]    #The guess
            resultArr[i,j]+=1

print(resultArr)
