import cv2
import pytesseract


def ocr(image, platePath):

	plate = cv2.imread(platePath)
	text = pytesseract.image_to_string(plate , lang='eng')
	print(text)

	cv2.putText(image,text, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, 255,2)
	