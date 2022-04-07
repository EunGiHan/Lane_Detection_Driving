# -*- coding: utf-8 -*-
import cv2
import numpy as np

def on_trackbar(x):
    pass

img = cv2.imread("../cam/50-1.png")

if img is None:
    print("Image load failed!")
    exit(1)

blur = cv2.GaussianBlur(img, (3, 3), 0)
hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

cv2.imshow("src", img)
cv2.namedWindow("hsv")
cv2.createTrackbar("low_H", "hsv", 0, 179, on_trackbar)
cv2.createTrackbar("low_S", "hsv", 0, 255, on_trackbar)
cv2.createTrackbar("low_V", "hsv", 26, 255, on_trackbar)

cv2.createTrackbar("high_H", "hsv", 179, 179, on_trackbar)
cv2.createTrackbar("high_S", "hsv", 88, 255, on_trackbar)
cv2.createTrackbar("high_V", "hsv", 120, 255, on_trackbar)

while cv2.waitKey(1) != 27:
    low_H = cv2.getTrackbarPos("low_H", "hsv")
    low_S = cv2.getTrackbarPos("low_S", "hsv")
    low_V = cv2.getTrackbarPos("low_V", "hsv")
    
    high_H = cv2.getTrackbarPos("high_H", "hsv")
    high_S = cv2.getTrackbarPos("high_S", "hsv")
    high_V = cv2.getTrackbarPos("high_V", "hsv")
    
    low = np.array([low_H, low_S, low_V])
    high = np.array([high_H, high_S, high_V])
    
    bin = cv2.inRange(hsv, low, high)
    cv2.imshow("hsv", bin)