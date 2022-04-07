#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2, rospy, math
import numpy as np

from xycar_motor.msg import xycar_motor
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# 명도차 함수 상수 값
low = np.array([0, 0, 100])         # HSV 하한값
high = np.array([131, 255, 255])    # HSV 상한값
width, height = 640, 480            # 영상의 크기
mid_l, mid_r = 200, 440             # 영상의 중간 좌우 픽셀 값
roi_w, roi_h = 200, 20              # 좌우 차선 ROI 영역의 크기 값
box_w, box_h = 15, 20               # 차선 체크박스 크기
offset = 430                        # ROI 창의 높이
pixel_threshold = int(0.6 * box_w * box_h)   # 차선 체크박스 임계값


first_find_pos = 320
second_find_pos = 320

# 이동평균 필터 상수
k = 0                               # k번 째 수 의미
preAvg = 0                          # 이전의 평균 값
N = 10                              # 슬라이딩 윈도우 크기
c_buf = np.zeros(N + 1)             # 슬라이딩 윈도우

# 이미지 처리 함수
def process_image(cv_img):
    global roi_h, offset, low, high
    show = cv_img.copy()
    blur = cv2.GaussianBlur(cv_img,(3, 3), 0)       # 가우시안 필터
    hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)      # HSV 변환
    hsv_roi = hsv[offset : offset + roi_h, :].copy()       # HSV의 ROI 추출
    bin_roi = cv2.inRange(hsv_roi, low, high)
    return show, bin_roi

# 차선 좌표 검출 함수
def get_line_pos(bin_roi):
    global box_w, box_h, mid_l, mid_r, pixel_threshold, width, first_find_pos, second_find_pos
    l_pos, r_pos = box_w / 2, width - box_w /2

    for i in range(width+1):
        # 무조건 하나는 찾는다고 가정
        box = bin_roi[0:box_h + 1, i:i + box_w + 1]
        if cv2.countNonZero(box) > pixel_threshold:
            first_find_pos = i
            break

    for i in range(i+1, width+1):
        box = bin_roi[0:box_h + 1, i:i + box_w + 1]
        if cv2.countNonZero(box) > pixel_threshold:
            second_find_pos = i
            l_pos = first_find_pos
            r_pos = second_find_pos
            print("find both")
            break
        if i == width:  # 차선 하나만 인식함
            if l_pos < 5:   # 이전 왼쪽 위치가 5 이하. 왼쪽 차선 잃음
                r_pos = first_find_pos
                l_pos = 0
                print("find right")
            elif r_pos > 635:   # 이전 오른쪽 위치가 635 이상. 오른쪽 차선 잃음
                l_pos = first_find_pos
                r_pos = 640
                print("find left")
    
    c_pos = (l_pos + r_pos) / 2     # 중간 픽셀 계산
    return l_pos, r_pos, c_pos

# angle 튜닝 함수
def generate_angle(c_pos):
    global width
    delta_pixel = c_pos - width/2
    angle = 0
    if abs(delta_pixel) < 5:        # angle 조건문
        angle = delta_pixel * 0.4
    else:
        angle = delta_pixel * 0.7
    return delta_pixel, angle

# 이동 평균 필터
def movAvgFilter(c_pos):
    global k, preAvg, c_buf, N
    if k == 0:
        c_buf = c_pos*np.ones(N + 1)
        k, preAvg = 1, c_pos
        
    for i in range(0, N):
        c_buf[i] = c_buf[i + 1]
    
    c_buf[N] = c_pos
    avg = preAvg + (c_pos - c_buf[0]) / N
    preAvg = avg
    return int(round(avg))
        
# 스티어 그림 함수
def draw_steer(show, angle):
    angle = - angle
    global width, height, arrow_pic
    arrow_pic = cv2.imread('/home/jrvm/xycar_ws/src/week7/hough_drive/src/steer_arrow.png', cv2. IMREAD_COLOR)

    origin_Height, origin_Width = arrow_pic.shape[:2]
    steer_wheel_center = origin_Height * 0.74
    arrow_Height = height/2
    arrow_Width = (arrow_Height * 462)/728

    matrix = cv2.getRotationMatrix2D((origin_Width/2, steer_wheel_center), (angle) * 2.5, 0.7)

    arrow_pic = cv2. warpAffine(arrow_pic, matrix, (origin_Width+60, origin_Height))
    arrow_pic = cv2.resize(arrow_pic, dsize=(arrow_Width, arrow_Height), interpolation=cv2.INTER_AREA)
    
    gray_arrow = cv2.cvtColor(arrow_pic, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_arrow, 1, 255, cv2.THRESH_BINARY_INV)

    arrow_roi = show[arrow_Height: height, (width/2 - arrow_Width/2) : (width/2 + arrow_Width/2)]
    arrow_roi = cv2.add(arrow_pic, arrow_roi, mask=mask)
    res = cv2.add(arrow_roi, arrow_pic)
    show[(height - arrow_Height): height, (width/2 - arrow_Width/2): (width/2 + arrow_Width/2)] = res

# 박스, 그리기 함수
def draw_box(show, bin_roi, l_pos, r_pos, c_pos):
    global box_w, box_h, width, offset
    show_bin = cv2.cvtColor(bin_roi, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(show, (l_pos - box_w/2, offset), (l_pos + box_w/2, offset + box_h), (0, 255, 0), 1)
    cv2.rectangle(show, (r_pos - box_w/2, offset), (r_pos + box_w/2, offset + box_h), (0, 255, 0), 1)
    cv2.rectangle(show, (c_pos - box_w/2, offset), (c_pos + box_w/2, offset + box_h), (0, 255, 0), 1)
    cv2.rectangle(show, (width/2 - box_w/2, offset), (width/2 + box_w/2, offset + box_h), (0, 0, 255), 1)
    
    cv2.rectangle(show_bin, (l_pos - box_w/2, 0), (l_pos + box_w/2, box_h), (0, 255, 0), 1)
    cv2.rectangle(show_bin, (r_pos - box_w/2, 0), (r_pos + box_w/2, box_h), (0, 255, 0), 1)
    cv2.rectangle(show_bin, (c_pos - box_w/2, 0), (c_pos + box_w/2, box_h), (0, 255, 0), 1)
    cv2.rectangle(show_bin, (width/2 - box_w/2, 0), (width/2 + box_w/2, box_h), (0, 0, 255), 1)
    
    dst = cv2.vconcat([show, show_bin])
    return dst

# 모터 퍼블리시 함수
def pub_motor(angle, speed):
    global pub
    global motor_control
    motor_control.angle = angle
    motor_control.speed = speed

    pub.publish(motor_control)

# 콜백 함수 선언
def img_callback(data):
    global cv_img
    cv_img = bridge.imgmsg_to_cv2(data, "bgr8")
    
# cv 브릿지
bridge = CvBridge()
cv_img = np.empty(shape=[0])

# 기본 로스 함수
rospy.init_node('line_follow')
rospy.Subscriber("/usb_cam/image_raw/", Image, img_callback)
pub = rospy.Publisher("xycar_motor", xycar_motor, queue_size=1)
motor_control = xycar_motor()

while not rospy.is_shutdown():
    if cv_img.size != (width*height*3):
        continue
    
    show, bin_roi = process_image(cv_img)
    l_pos, r_pos, c_pos = get_line_pos(bin_roi)
    c_avg = movAvgFilter(c_pos)
    delta_pixel1, angle1 = generate_angle(c_pos)
    delta_pixel2, angle2 = generate_angle(c_avg)
    pub_motor(angle2, 30)
    
    # draw_steer(show, angle)
    # dst = draw_box(show, bin_roi, l_pos, r_pos, c_pos)
    dst = draw_box(show, bin_roi, l_pos, r_pos, c_avg)
    print("픽셀 차이: {}, 필터 픽셀 차이: {}".format(delta_pixel1, delta_pixel2,))
    cv2.imshow("dst", dst)
    cv2.waitKey(1)
    