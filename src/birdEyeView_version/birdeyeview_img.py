#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2, math
from cv2 import threshold
import numpy as np

def trackbar_callback(x):
    pass

img_src = cv2.imread("../cam/src.png")
warp_img_width = 320    # 와핑 가로
warp_img_height = 240
warp_dst = np.array([
            [0, 0],
            [0, warp_img_height],
            [warp_img_width, 0],
            [warp_img_width, warp_img_height]
        ], dtype = np.float32)

if img_src is None:
    print("Image load failed!")
    exit(1)

# cv2.imshow("src", img)
cv2.namedWindow("bev")
cv2.createTrackbar("left_top_x", "bev", 180, 640, trackbar_callback)
cv2.createTrackbar("right_top_x", "bev", 460, 640, trackbar_callback)
cv2.createTrackbar("left_bottom_x", "bev", 20, 640, trackbar_callback)
cv2.createTrackbar("right_bottom_x", "bev", 620, 640, trackbar_callback)
cv2.createTrackbar("top_y", "bev", 270, 480, trackbar_callback)
cv2.createTrackbar("bottom_y", "bev", 370, 480, trackbar_callback)

cv2.createTrackbar("threshold", "bev", 100, 200, trackbar_callback)
cv2.createTrackbar("minLineLength", "bev", 50, 100, trackbar_callback)
cv2.createTrackbar("maxLineGap", "bev", 10, 100, trackbar_callback)

while True:
    img = img_src
    color_warp_area = np.zeros_like(img).astype(np.uint8)

    left_top = [cv2.getTrackbarPos("left_top_x", "bev"), cv2.getTrackbarPos("top_y", "bev")]
    left_bottom = [cv2.getTrackbarPos("left_bottom_x", "bev"), cv2.getTrackbarPos("bottom_y", "bev")]
    right_top = [cv2.getTrackbarPos("right_top_x", "bev"), cv2.getTrackbarPos("top_y", "bev")]
    right_bottom = [cv2.getTrackbarPos("right_bottom_x", "bev"), cv2.getTrackbarPos("bottom_y", "bev")]

    thre = cv2.getTrackbarPos("threshold", "bev")
    minLineLength = cv2.getTrackbarPos("minLineLength", "bev")
    maxLineGap = cv2.getTrackbarPos("maxLineGap", "bev")

    pts = np.array([left_top, left_bottom, right_bottom, right_top])
    color_warp_area = cv2.fillPoly(color_warp_area, np.int_([pts]), (0, 255, 0))
    img = cv2.addWeighted(img, 1, color_warp_area, 0.3, 0)
    cv2.imshow("bev", img)

    warp_src = np.array([left_top, left_bottom, right_top, right_bottom], dtype = np.float32)
    perspec_mat = cv2.getPerspectiveTransform(warp_src, warp_dst)
    warp_img = cv2.warpPerspective(img, perspec_mat, (warp_img_width, warp_img_height), flags=cv2.INTER_LINEAR)

    blur = cv2.GaussianBlur(warp_img,(3, 3), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(gray, 100, 200)
    lines = cv2.HoughLinesP(edge, 1, math.pi/180, thre, None, minLineLength, maxLineGap) # 확인
    edge = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)

    if lines is not None:
        for line in lines:
            # cv2.addText(edge, "{0} / {1} // {2} / {3}".format(line[0][0], line[0][1], line[0][2], line[0][3]), (10, 10))
            cv2.line(edge, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (0, 0, 255), 2, 0)
    cv2.imshow("warp_img", edge)

    if cv2.waitKey(1) == 27:
        break