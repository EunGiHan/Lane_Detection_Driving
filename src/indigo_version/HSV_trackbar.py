# -*- coding: utf-8 -*-
import cv2, math, random
import numpy as np

def draw_rectangle(img, lpos, rpos, offset=0):
    center = (lpos + rpos) / 2 -10

    cv2.rectangle(img, (lpos - 5, 15 + offset),
                    (lpos + 5, 25 + offset),
                    (0, 255, 0), 2)
    cv2.rectangle(img, (rpos - 5, 15 + offset),
                    (rpos + 5, 25 + offset),
                    (0, 255, 0), 2)
    cv2.rectangle(img, (center - 5, 15 + offset),
                    (center + 5, 25 + offset),
                    (0, 255, 0), 2)
    cv2.rectangle(img, (640 / 2 - 5, 15 + offset),
                    (640 / 2 + 5, 25 + offset),
                    (0, 0, 255), 2)
    return img

def on_trackbar(x):
    pass

def divide_left_right(lines):
    low_slope_threshold = 0
    high_slope_threshold = 10

    # calculate slope & filtering with threshold
    slopes = []
    new_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]

        if x2 - x1 == 0:
            slope = 0
        else:
            slope = float(y2 - y1) / float(x2 - x1)

        if abs(slope) > low_slope_threshold and abs(slope) < high_slope_threshold:
            slopes.append(slope)
            new_lines.append(line[0])

    # divide lines left to right
    left_lines = []
    right_lines = []
    middle_thresh = 0
    for j in range(len(slopes)):
        Line = new_lines[j]
        slope = slopes[j]

        x1, y1, x2, y2 = Line

        if (slope < 0) and (x2 < 640 / 2 - middle_thresh):
            left_lines.append([Line.tolist()])
        elif (slope > 0) and (x1 > 640 / 2 + middle_thresh):
            right_lines.append([Line.tolist()])

    return left_lines, right_lines

# get average m, b of lines
def get_line_params(lines):
    # sum of x, y, m
    x_sum = 0.0
    y_sum = 0.0
    m_sum = 0.0

    size = len(lines)
    if size == 0:
        return 0, 0

    for line in lines:
        x1, y1, x2, y2 = line[0]

        x_sum += x1 + x2
        y_sum += y1 + y2
        m_sum += float(y2 - y1) / float(x2 - x1)

    x_avg = x_sum / (size * 2)
    y_avg = y_sum / (size * 2)
    m = m_sum / size
    b = y_avg - m * x_avg

    return m, b

def draw_lines(img, lines):
    for line in lines:
        x1, y1, x2, y2 = line[0]
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        img = cv2.line(img, (x1, y1 + 380), (x2, y2 + 380), color, 2)
    return img

# get lpos, rpos
def get_line_pos(img, lines, left=False, right=False):

    m, b = get_line_params(lines)

    if m == 0 and b == 0:
        if left:
            pos = -1
        if right:
            pos = 640 + 1
    else:
        y = 40 / 2
        pos = (y - b) / m

        b += 380
        x1 = (480 - b) / float(m)
        x2 = ((480 / 2) - b) / float(m)

        cv2.line(img, (int(x1), 480), (int(x2), (480 / 2)), (255, 0, 0), 3)

    return img, int(pos)

imgs = [0, 0, 0, 0, 0, 0, 0, 0, 0]
num = 0
for i in range(1, 10):
    file_name = "../cam/50-" + str(i) + ".png"
    img = cv2.imread(file_name)
    imgs[i-1] = img

cv2.namedWindow("hsv")
cv2.createTrackbar("low_H", "hsv", 0, 179, on_trackbar)
cv2.createTrackbar("low_S", "hsv", 0, 255, on_trackbar)
cv2.createTrackbar("low_V", "hsv", 0, 255, on_trackbar)

cv2.createTrackbar("high_H", "hsv", 179, 179, on_trackbar)
cv2.createTrackbar("high_S", "hsv", 88, 255, on_trackbar)
cv2.createTrackbar("high_V", "hsv", 120, 255, on_trackbar)

while True:
    low_H = cv2.getTrackbarPos("low_H", "hsv")
    low_S = cv2.getTrackbarPos("low_S", "hsv")
    low_V = cv2.getTrackbarPos("low_V", "hsv")
    
    high_H = cv2.getTrackbarPos("high_H", "hsv")
    high_S = cv2.getTrackbarPos("high_S", "hsv")
    high_V = cv2.getTrackbarPos("high_V", "hsv")

    low = np.array([low_H, low_S, low_V])
    high = np.array([high_H, high_S, high_V])

    hsv = cv2.cvtColor(imgs[num], cv2.COLOR_BGR2HSV)
    # gray = cv2.cvtColor(hsv, cv2.COLOR_BGR2GRAY)
    bin = cv2.inRange(hsv, low, high)
    blur = cv2.GaussianBlur(bin, (5, 5), 0)
    edge_img = cv2.Canny(np.uint8(blur), 140, 70)
    roi = edge_img[377: 377 + 40, 0: 640]
    all_lines = cv2.HoughLinesP(roi, 1, math.pi / 180, 30, 30, 10)

    if all_lines is None:
        lpos = 0
        rpos = 640
        continue
    left_lines, right_lines = divide_left_right(all_lines)

    # get center of lines
    imgs[num], lpos = get_line_pos(imgs[num], left_lines, left=True)
    imgs[num], rpos = get_line_pos(imgs[num], right_lines, right=True)

    # draw lines
    imgs[num] = draw_lines(imgs[num], left_lines)
    imgs[num] = draw_lines(imgs[num], right_lines)
    imgs[num] = cv2.line(imgs[num], (230, 235), (410, 235), (255, 255, 255), 2)

    # draw rectangle
    imgs[num] = draw_rectangle(imgs[num], lpos, rpos, offset=380)

    cv2.imshow("hsv", bin)
    cv2.imshow("img", imgs[num])

    if cv2.waitKey(1) == 27:
        # print(num)
        # num += 1
        break