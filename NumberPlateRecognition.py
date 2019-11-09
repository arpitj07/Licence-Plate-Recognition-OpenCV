import cv2
import numpy as np 
import imutils
import pytesseract
import os
import preprocess
import OCR



#########################DEFINE VARIABLES####################################################
imagepath = 'Images/car5'
platePath =  'Output/7.png'
NumberPlate = None
idx = 7 
#############################################################################################



# READ IMAGE 
image = cv2.imread(imagepath + '.jpg')
image = cv2.resize(image , (300,300))

#EXTRACT VALUE FROM IMAGE
gray = preprocess.extractValue(image)
gray = cv2.bilateralFilter(gray, 11,17,17)


#FIND EDGES
edged = cv2.Canny(gray, 170,200)


#FIND CONTOURS
cnts = cv2.findContours(edged,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0]


#SORTING THE CONTOURS TO GET TOP 30
cnts = sorted(cnts , key= cv2.contourArea , reverse=True)[:30]

#LOOP OVER TOP 30 CONTOUR AND FETCH THE ONE WITH 4 CORNERS
for c in cnts:
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c , 0.02*peri , True)

	if len(approx)==4:
		NumberPlate=approx
		[x,y,w,h] = cv2.boundingRect(c)
		new_image = image[y:y+h, x:x+w]
		cv2.imwrite('Output/'+ str(idx)+'.png', new_image)
		idx+=1
		break

#DRAW CONTOUR OVER THE PLATE
cv2.drawContours(image, [NumberPlate], -1, (0,0,255), 3)

#PRINT THE TEXT ON THE PLATE
OCR.ocr(image, platePath)

cv2.imshow('car' , image)
cv2.imshow("Plate" ,new_image)
cv2.waitKey(0)
cv2.destroyAllWindows()