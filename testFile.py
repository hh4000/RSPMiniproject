import cv2 as cv
import numpy as np

#Hans path (comment out if not Hans)
#path = r"C:\Users\hansh\OneDrive - Aalborg Universitet\Programmer\3. semester\RSP\Miniproject\Training set\1.jpg"

#Hans path (comment out if not Hans)
#path = ---- ADD PATH HERE ----

testImg = cv.imread(path, 1)
cv.imshow("picture",testImg)
cv.waitKey(0)