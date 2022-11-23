from matplotlib import pyplot as plt
import numpy as np
import cv2
import imutils

#loading our base image
file_path = r'C:\Users\silas\Desktop\Robotics 3. semester\Image processing miniproject\RSPMiniproject\Training set\31.jpg'
img = cv2.imread(file_path,1)
img_gray = cv2.imread(file_path,0)
#Loading all the templates for each type of tile
Grass = cv2.imread(r'C:\Users\silas\Desktop\Robotics 3. semester\Image processing miniproject\Grass temp.jpg',0)
Forest = cv2.imread(r'C:\Users\silas\Desktop\Robotics 3. semester\Image processing miniproject\Forest temp.jpg',0)
Ocean = cv2.imread(r'C:\Users\silas\Desktop\Robotics 3. semester\Image processing miniproject\Ocean temp.jpg',0)
Swamp = cv2.imread(r'C:\Users\silas\Desktop\Robotics 3. semester\Image processing miniproject\Swamp temp.jpg',0)
Wheat = cv2.imread(r'C:\Users\silas\Desktop\Robotics 3. semester\Image processing miniproject\Wheat temp.jpg',0)
Mountain = cv2.imread(r'C:\Users\silas\Desktop\Robotics 3. semester\Image processing miniproject\Mountain temp.jpg',0)

def sharpen(img):
    filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    img=cv2.filter2D(img,-1,filter)
    return img

def blur(img):
    kernel = np.ones((5,5),np.float32)/25
    img = cv2.filter2D(img,-1,kernel)
    return img

def rotate(template):
    r1 = template
    r2 = cv2.rotate(r1,cv2.ROTATE_90_CLOCKWISE)
    r3 = cv2.rotate(r1,cv2.ROTATE_90_COUNTERCLOCKWISE)
    r4 = cv2.rotate(r1,cv2.ROTATE_180)
    rotations = np.array([r1,r2,r3,r4])
    return rotations


def templateMatch(img, template, thresh):
    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)

    loc = np.where( res >= thresh)  
    #Outputs 2 arrays. Combine these arrays to get x,y coordinates - take x from one array and y from the other.

    #Reminder: ZIP function is an iterator of tuples where first item in each iterator is paired together,
    #then the second item and then third, etc. 

    for pt in zip(*loc[::-1]):   #-1 to swap the values as we assign x and y coordinate to draw the rectangle. 
    #Draw rectangle around each object. We know the top left (pt), draw rectangle to match the size of the template image.
        cv2.rectangle(img, pt, (pt[0] + 30, pt[1] + 30), (0, 0, 255), 2)  #Red rectangles with thickness 2. 
    #cv2.imshow("Matched image", img2)
    #cv2.waitKey()
    #cv2.destroyAllWindows()


match1 = rotate(Grass)
match2 = rotate(Forest)
match3 = rotate(Ocean)
match4 = rotate(Swamp)
match5 = rotate(Wheat)
match6 = rotate(Mountain)

#[templates.append(a) for a in rotate(Forest)]
#[templates.append(a) for a in rotate(Ocean)]
#[templates.append(a) for a in rotate(Swamp)]
#[templates.append(a) for a in rotate(Wheat)]
#[templates.append(a) for a in rotate(Mountain)]


for x in range(0,4):
    templateMatch(img,match1[x],0.75)
    templateMatch(img,match2[x],0.75)
    templateMatch(img,match3[x],0.75)
    templateMatch(img,match4[x],0.75)
    templateMatch(img,match5[x],0.6)
    templateMatch(img,match6[x],0.8)



cv2.imshow("Matched image", img)
cv2.waitKey()
cv2.destroyAllWindows()



