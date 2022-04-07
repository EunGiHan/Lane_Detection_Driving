#!/usr/bin/env python
#-*- coding:utf-8 -*-

import cv2
import rospy
import math
import numpy as np

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class BirdEyeView:
    def __init__(self):
        rospy.Subscriber("/usb_cam/image_raw/", Image, self.img_callback)
        self.bridge = CvBridge()
        self.frame = np.empty(shape=[0])

        self.img_width = 640    # 원본 이미지 가로
        self.img_height = 480    # 원본 이미지 세로

        ### xycar 캘리브레이션 보정값들
        self.cam_matrix = np.array([
            [422.037858, 0.0, 245.895397],
            [0.0, 435.589734, 163.625535],
            [0.0, 0.0, 1.0]])  # 입력 카메라 내부 행렬
        self.dist_coeffs = np.array([-0.289296, 0.061035, 0.001786, 0.015238, 0.0]) # 왜곡 계수의 입력 벡터
        self.new_cam_matrix, self.valid_pix_ROI = \
            cv2.getOptimalNewCameraMatrix(self.cam_matrix, self.dist_coeffs,(self.img_width, self.img_height), 1, (self.img_width, self.img_height))

        ### warp 관련 값 정의 -> 원본의 절반 크기로 birds-eye view로 펼침
        self.warp_img_width = 320    # 와핑 가로
        self.warp_img_height = 240   # 와핑 세로

        self.left_top, self.left_bottom, self.right_top, self.right_bottom = [0, 0], [0, 0], [0, 0], [0, 0]
        
        self.warp_src = np.array([self.left_top, self.left_bottom, self.right_top, self.right_bottom], dtype = np.float32)  # 위 왼쪽 / 아래 왼쪽 / 위 오른쪽 / 아래 오른쪽

        self.warp_dst = np.array([
            [0, 0],
            [0, self.warp_img_height],
            [self.warp_img_width, 0],
            [self.warp_img_width, self.warp_img_height]
        ], dtype = np.float32)

        self.l_start = 0
        self.l_end = 255

        while self.frame.size != (640*480*3):
            continue

        # cv2.imshow("src", self.frame)
        cv2.namedWindow("bev")
        cv2.createTrackbar("left_top_x", "bev", 0, 640, self.trackbar_callback)
        cv2.createTrackbar("left_top_y", "bev", 0, 480, self.trackbar_callback)
        cv2.createTrackbar("left_bottom_x", "bev", 0, 640, self.trackbar_callback)
        cv2.createTrackbar("left_bottom_y", "bev", 0, 480, self.trackbar_callback)
        cv2.createTrackbar("right_top_x", "bev", 0, 640, self.trackbar_callback)
        cv2.createTrackbar("right_top_y", "bev", 0, 480, self.trackbar_callback)
        cv2.createTrackbar("right_bottom_x", "bev", 0, 640, self.trackbar_callback)
        cv2.createTrackbar("right_bottom_y", "bev", 0, 480, self.trackbar_callback)
        
    def poly(self):
        color_warp_area = np.zeros_like(self.frame).astype(np.uint8)
        self.value_set()
        pts = np.array([self.left_top, self.left_bottom, self.right_top, self.right_bottom])
        color_warp_area = cv2.fillPoly(color_warp_area, np.int_([pts]), (0, 255, 0))
        self.frame = cv2.addWeighted(self.frame, 1, color_warp_area, 0.3, 0)
        cv2.imshow("bev", self.frame)

    def start(self):
        self.calibrate_img()
        self.warp_image()
        self.value_set()
        self.binalization()
        cv2.imshow("src", self.frame)
        # cv2.imshow("warp_img", self.warp_img)
        # cv2.imshow("lane", self.lane)

        if cv2.waitKey(1) == 27:
            return

    def img_callback(self, msg):
        self.frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def calibrate_img(self):
        self.tf_img = cv2.undistort(self.frame, self.cam_matrix, self.dist_coeffs, None, self.new_cam_matrix)
            # 구한 카메라 보정 행렬을 이용해 이미지를 보정함
        x, y, w, h = self.valid_pix_ROI
        self.tf_img = self.tf_img[y:y+h, x:x+w]   # 보정 이미지 중에서 이미지의 원래 위치(?)에 맞게 잘라내기
        cv2.resize(self.tf_img, (self.img_width, self.img_height))

    def warp_image(self):
        self.perspec_mat = cv2.getPerspectiveTransform(self.warp_src, self.warp_dst)
        self.perspec_mat_inv = cv2.getPerspectiveTransform(self.warp_dst, self.warp_src)
        self.warp_img = cv2.warpPerspective(self.tf_img, self.perspec_mat, (self.warp_img_width, self.warp_img_height), flags=cv2.INTER_LINEAR)
        cv2.imshow("warp_img", self.warp_img)

    def binalization(self):
        blur = cv2.GaussianBlur(self.warp_img, (5, 5), 0)
        _, self.L, _ = cv2.split(cv2.cvtColor(blur, cv2.COLOR_BGR2HLS))
        _, self.lane = cv2.threshold(self.L, self.l_start, self.l_end, cv2.THRESH_BINARY)  # 검은색으로 바꿀 것
        cv2.imshow("L", self.L)
        cv2.imshow("lane", self.lane)

    def value_set(self):
        self.left_top = [cv2.getTrackbarPos("left_top_x", "bev"), cv2.getTrackbarPos("left_top_y", "bev")]
        self.left_bottom = [cv2.getTrackbarPos("left_bottom_x", "bev"), cv2.getTrackbarPos("left_bottom_y", "bev")]
        self.right_top = [cv2.getTrackbarPos("right_top_x", "bev"), cv2.getTrackbarPos("right_top_y", "bev")]
        self.right_bottom = [cv2.getTrackbarPos("right_bottom_x", "bev"), cv2.getTrackbarPos("right_bottom_y", "bev")]

    def trackbar_callback(self, data):
        pass


def main():
    rospy.init_node("birdeyeview", anonymous=False)
    bdv = BirdEyeView()

    while not rospy.is_shutdown():
        while bdv.frame.size != (640*480*3):
            continue
        # bdv.start()
        bdv.poly()

if __name__=='__main__':
    main()