import numpy as np
import cv2
from collections import deque


img = cv2.imread(r'C:\Users\silas\Desktop\Robotics 3. semester\Billedbehandling kursus (ROB3)\Notes for Billedbehandling course\Python implementations for exam\Lecture exercises (perception)\shapes.png',0)


img2 = np.array([['s', 'w', 'f', 'f', 'f'],
                ['w', 'w', 'f', 'f', 'f'], 
                ['m', 'w', 'H', 'f', 'o'], 
                ['w', 'f', 'f', 'f', 'o'], 
                ['o', 'o', 'o', 'f', 'o']])

#This function burns a single section and gives it a single number id
def ignite_pixel(image,coordinate,id, char):
    y,x = coordinate
    burn_queue = deque()
    something_burned = False
    #If the image value is equal to what we are looking for, the character, then it adds it to the burn queue
    if image[y,x] == char:
        burn_queue.append((y,x))
    #While the burn queue still has something to burn it will loop
    while len(burn_queue) > 0:
        #Here the coordinate is burned by giving it the "id" value
        current_coordinate = burn_queue.pop() #This removes the last coordinate while saving it to current_coordinate
        y,x = current_coordinate
        image[y,x] = id
        #This is used to change the id after all is burned
        something_burned = True

        #These if-statements check the tiles around the current tile
        if x+1 < image.shape[1] and image[y,x+1] == char:
            burn_queue.append((y,x+1))
        if y+1 < image.shape[0] and image[y+1,x] == char:
            burn_queue.append((y+1,x))
        if x-1 >= 0 and image[y,x-1] == char:
            burn_queue.append((y,x-1))
        if y-1 >= 0 and image[y-1,x] == char:
            burn_queue.append((y-1,x))
 
    if something_burned:
        return id +1
    return id

def grassfire(image):
    next_id = 0
    #Takes all the different characters from the "image" 
    tiles = np.unique(image)
    #checks all type of characters
    for char in tiles:
        #checks all "pixels"
        for y, row in enumerate(image):
            for x, pixel in enumerate(row):
                next_id = ignite_pixel(image, (y,x), next_id, char)
    return image
    
    

grassfire(img2)
print(img2)


