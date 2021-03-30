import cv2
import numpy as np


frameWidth = 640
frameHeight = 480
#cap = cv2.VideoCapture(1)
#cap.set(3, frameWidth)
#cap.set(4, frameHeight)
#cap.set(10,150)

def preProcessing(img):
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)    
    imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
    imgCanny = cv2.Canny(imgBlur,200,200)
    #kernel = np.ones((5,5))
    #imgDial = cv2.dilate(imgCanny,kernel,iterations=2)
    #imgThres = cv2.erode(imgDial,kernel,iterations=1)

    return imgCanny

def getContours(img):
    biggest = np.array([])
    maxArea = 0
    contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area>5000:
            #cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)
            if area >maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
    #cv2.drawContours(imgContour, biggest, -1, (255, 0, 0), 20)
    return biggest


img = cv2.imread("Resources/MILK0158.png")

'''
while True:
    success, img = cap.read()
    cv2.imshow("Result",img)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break
'''
imgThres = preProcessing(img)
biggest = getContours(imgThres)

print(biggest.shape)

cv2.imshow("Original", imgThres)
cv2.waitKey(0)