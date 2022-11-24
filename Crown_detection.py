from matplotlib import pyplot as plt
import numpy as np
import cv2
import imutils

#loading our base image
file_path = r'C:\Users\silas\Desktop\Robotics 3. semester\Image processing miniproject\RSPMiniproject\Training set\36.jpg'
img = cv2.imread(file_path,1)
img_gray = cv2.imread(file_path,0)
#Loading all the templates for each type of tile
Grass = cv2.imread(r'C:\Users\silas\Desktop\Robotics 3. semester\Image processing miniproject\Grass temp.jpg',0)
Forest = cv2.imread(r'C:\Users\silas\Desktop\Robotics 3. semester\Image processing miniproject\Forest temp.jpg',0)
Ocean = cv2.imread(r'C:\Users\silas\Desktop\Robotics 3. semester\Image processing miniproject\Ocean temp.jpg',0)
Swamp = cv2.imread(r'C:\Users\silas\Desktop\Robotics 3. semester\Image processing miniproject\Swamp temp.jpg',0)
Wheat = cv2.imread(r'C:\Users\silas\Desktop\Robotics 3. semester\Image processing miniproject\Wheat temp.jpg',0)
Mountain = cv2.imread(r'C:\Users\silas\Desktop\Robotics 3. semester\Image processing miniproject\Mountain temp.jpg',0)


def rotate(template):
    #A function that rotates each template 3 times and saves it as a numpy array
    r1 = template
    r2 = cv2.rotate(r1,cv2.ROTATE_90_CLOCKWISE)
    r3 = cv2.rotate(r1,cv2.ROTATE_90_COUNTERCLOCKWISE)
    r4 = cv2.rotate(r1,cv2.ROTATE_180)
    rotations = np.array([r1,r2,r3,r4],dtype=object)
    return rotations


def NMS(boxes, overlapThresh = 0.4):
    # Return an empty list, if no boxes given
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]  # x coordinate of the top-left corner
    y1 = boxes[:, 1]  # y coordinate of the top-left corner
    x2 = boxes[:, 2]  # x coordinate of the bottom-right corner
    y2 = boxes[:, 3]  # y coordinate of the bottom-right corner
    # Compute the area of the bounding boxes and sort the bounding
    # Boxes by the bottom-right y-coordinate of the bounding box
    areas = (x2 - x1 + 1) * (y2 - y1 + 1) # We add 1, because the pixel at the start as well as at the end counts
    # The indices of all boxes at start. We will redundant indices one by one.
    indices = np.arange(len(x1))
    for i,box in enumerate(boxes):
        # Create temporary indices  
        temp_indices = indices[indices!=i]
        # Find out the coordinates of the intersection box
        xx1 = np.maximum(box[0], boxes[temp_indices,0])
        yy1 = np.maximum(box[1], boxes[temp_indices,1])
        xx2 = np.minimum(box[2], boxes[temp_indices,2])
        yy2 = np.minimum(box[3], boxes[temp_indices,3])
        # Find out the width and the height of the intersection box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / areas[temp_indices]
        # if the actual boungding box has an overlap bigger than treshold with any other box, remove it's index  
        if np.any(overlap) > overlapThresh:
            indices = indices[indices != i]
    #return only the boxes at the remaining indices
    return boxes[indices].astype(int)

#Create a global array that can be added to for each use of the templateMatch function
box = np.array([[0,0,0,0], [0,0,0,0]],dtype=object)

def templateMatch(img, template, thresh):
    #cv2 does template matchtin 
    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    #It is then saved in terms of a threshold
    loc = np.where( res >= thresh)  
    #Outputs 2 arrays. Combine these arrays to get x,y coordinates - take x from one array and y from the other
    global box
    #Reminder: ZIP function is an iterator of tuples where first item in each iterator is paired together,
    #then the second item and then third, etc. 
    for pt in zip(*loc[::-1]):   #-1 to swap the values as we assign x and y coordinate to draw the rectangle. 
        #The coordinates are then saved in the global box array for use later
        box =np.append(box, [[pt[0], pt[1], (pt[0]+20), (pt[1]+20)]], axis=0)
    
#This function uses the boxes coordinates to draw the them on the image
def draw(boxes):
    for x in range(boxes.shape[0]):
        cv2.rectangle(img, (boxes[x,0], boxes[x,1]), (boxes[x,2], boxes[x,3]), (0, 0, 255), 1)

#Function that rotates all the templates and then does templatematching on all of them
def matching():
    global box
    match1 = rotate(Grass)
    match2 = rotate(Forest)
    match3 = rotate(Ocean)
    match4 = rotate(Swamp)
    match5 = rotate(Wheat)
    match6 = rotate(Mountain)

    for x in range(0,4):
        templateMatch(img,match1[x],0.75)
        templateMatch(img,match2[x],0.75)
        templateMatch(img,match3[x],0.75)
        templateMatch(img,match4[x],0.75)
        templateMatch(img,match5[x],0.6)
        templateMatch(img,match6[x],0.8)
    #I had to do this because of the way that I set up the variable, since I can't append before I give the correct size
    box = np.delete(box,0,0)
    box = np.delete(box,0,0)

#Does all the functions
matching()
nms = NMS(box,0.4)
draw(nms)

#Then a function that finds the number of crowns in terms of a 5x5 grid of the entire picture
def finding_crown_pos(boxes):
    #I define an array that is 5x5 since the board can at max be 5x5
    num_crowns = np.zeros((5,5))
    #I then use the first two coordinates used to draw the crowns to determine how many crowns are in each tile
    for rows in range(boxes.shape[0]):
        x = int(boxes[rows,0]/100)
        y = int(boxes[rows,1]/100)
        num_crowns[y,x] = num_crowns[y,x] + 1
    return num_crowns

    

finding_crown_pos(nms)

cv2.imshow("Matched image", img)
cv2.waitKey()
cv2.destroyAllWindows()



