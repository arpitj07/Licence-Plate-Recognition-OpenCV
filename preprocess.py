import numpy as np 
import cv2 


structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))


def extractValue(image):

	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	hue, saturation , value = cv2.split(hsv)
	return value



def enhanceContrast(gray):

	topHat   = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT,   structuringElement)
	blackHat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, structuringElement)

	add  =  cv2.add(gray, topHat)
	diff = cv2.subtract(add, blackHat)
	return diff




def getThreshold(diff):

	blur   = cv2.GaussianBlur(diff , (5,5),0)
	thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
										      cv2.THRESH_BINARY_INV , 19,9)
	return blur, thresh
