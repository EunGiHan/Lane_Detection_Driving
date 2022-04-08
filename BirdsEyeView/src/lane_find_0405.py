#!/usr/bin/env python
# -*- coding: utf-8 -*-

from types import NoneType
import cv2
from cv2 import COLOR_GRAY2BGR
import rospy
import math
import numpy as np

from sensor_msgs.msg import Image
#from xycar_msgs.msg import xycar_motor
from cv_bridge import CvBridge

class LaneFind:
    def __init__(self):
        rospy.Subscriber("/usb_cam/image_raw/", Image, self.img_callback)
        #self.pub = rospy.Publisher('xycar_motor', xycar_motor, queue_size=1)

        self.bridge = CvBridge()
        self.frame = np.empty(shape=[0])

        self.img_width = 640    # 원본 이미지 가로
        self.img_height = 480    # 원본 이미지 세로

        self.warp_img_width = 320    # 와핑 가로
        self.warp_img_height = 240
        self.warp_img_mid = 160

        self.left_top = [180, 270]
        self.left_bottom = [20, 370]
        self.right_top = [460, 270]
        self.right_bottom = [620, 370]

        self.warp_src = np.array([self.left_top, self.left_bottom, self.right_top, self.right_bottom], dtype = np.float32)
        self.warp_dst = np.array([
            [0, 0],
            [0, self.warp_img_height],
            [self.warp_img_width, 0],
            [self.warp_img_width, self.warp_img_height]
        ], dtype = np.float32)
        self.perspec_mat = cv2.getPerspectiveTransform(self.warp_src, self.warp_dst)
        self.perspec_mat_inv = cv2.getPerspectiveTransform(self.warp_dst, self.warp_src)
        
        self.warp_img = None
        self.edge = None

        self.m_thr = 5
        self.deg_coeffi = 1
        self.dist_coeffi = 0.5

        self.lane_halfWidth = 400

        self.angle_kp = 0.45
        self.angle_ki = 0.0007
        self.angle_kd = 0.25
        self.angle_d_err = 0
        self.angle_p_err = 0
        self.angle_i_err = 0
        self.angle_max_i_err = 10
        self.angle_u = 0

        self.angle_err = 0
        self.steer_angle = 0

    def img_callback(self, data):
        self.frame = self.bridge.imgmsg_to_cv2(data, "bgr8")


    def process_warp_img(self):
        self.warp_img = cv2.warpPerspective(self.frame, self.perspec_mat, (self.warp_img_width, self.warp_img_height), flags=cv2.INTER_LINEAR)
        # self.edge = np.zeros_like(self.warp_img).astype(np.uint8)
        blur = cv2.GaussianBlur(self.warp_img,(3, 3), 0)
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        self.edge = cv2.Canny(gray, 100, 200)
        lines = cv2.HoughLinesP(self.edge, 1, math.pi/180, 100, None, 50, 10)
        self.edge = cv2.cvtColor(self.edge, cv2.COLOR_GRAY2BGR)

        left_lines = []
        right_lines = []
        left_found = False
        right_found = False

        if lines is None:
            return
    
        for line in lines:  ## 왼차선과 오른차선 구분
            line = line[0]
            ini_x, ini_y, fin_x, fin_y = line[0], line[1], line[2], line[3]
            if ini_x <= self.warp_img_mid and fin_x <= self.warp_img_mid:
                ## 왼쪽에 위치, 왼차선
                left_found = True
                left_lines.append([ini_x, ini_y])
                left_lines.append([fin_x, fin_y])
            elif ini_x > self.warp_img_mid and fin_x > self.warp_img_mid:
                ## 오른쪽에 위치, 오른차선
                right_found = True
                # right_lines.append(line)
                right_lines.append([ini_x, ini_y])
                right_lines.append([fin_x, fin_y])
            elif ini_x <= self.warp_img_mid and fin_x >= self.warp_img_mid:
                ## 걸쳐서 위치, 왼차선
                left_found = True
                # left_lines.append(line)
                left_lines.append([ini_x, ini_y])
                left_lines.append([fin_x, fin_y])
            elif ini_x >= self.warp_img_mid and fin_x <= self.warp_img_mid:
                ## 걸쳐서 위치, 오른차선
                right_found = True
                # right_lines.append(line)
                right_lines.append([ini_x, ini_y])
                right_lines.append([fin_x, fin_y])
            
            cv2.line(self.edge, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 1)
        
        left_fit = []
        right_fit = []
        ini_y, fin_y = self.warp_img_height, 0

        if left_found:
            ## 왼쪽 차선을 찾았다면
            left_lines = np.array(left_lines)
            left_fit = cv2.fitLine(left_lines, cv2.DIST_L2,0, 0.01, 0.01)
            left_m = left_fit[1] / left_fit[0]
            left_b = [left_fit[2], left_fit[3]]
            left_ini_x = ((ini_y - left_b[1]) / left_m) + left_b[0] if left_m != 0 else left_b[0]
            left_fin_x = ((fin_y - left_b[1]) / left_m) + left_b[0] if left_m != 0 else left_b[0]
            cv2.line(self.edge, (left_ini_x, ini_y), (left_fin_x, fin_y), (255, 0, 0), 2)

        if right_found:
            ## 오른쪽 차선을 찾았다면
            right_lines = np.array(right_lines)
            right_fit = cv2.fitLine(right_lines, cv2.DIST_L2,0, 0.01, 0.01)
            right_m = right_fit[1] / right_fit[0]
            right_b = [right_fit[2], right_fit[3]]
            right_ini_x = ((ini_y - right_b[1]) / right_m) + right_b[0] if right_m != 0 else right_b[0]
            right_fin_x = ((fin_y - right_b[1]) / right_m) + right_b[0] if right_m != 0 else right_b[0]
            cv2.line(self.edge, (right_ini_x, ini_y), (right_fin_x, fin_y), (255, 0, 0), 2)

        color_warp_area = np.zeros_like(self.frame).astype(np.uint8)
        pts = np.array([self.left_top, self.left_bottom, self.right_bottom, self.right_top])
        color_warp_area = cv2.fillPoly(color_warp_area, np.int_([pts]), (0, 255, 0))
        self.frame = cv2.addWeighted(self.frame, 1, color_warp_area, 0.3, 0)

        

        if left_found and right_found:
            m = (left_m + right_m) / 2
            mid = (left_ini_x + right_ini_x) / 2
            cv2.putText(self.edge, "both", (20, 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)
        elif (not left_found) and (not right_found):
            m = 0
            mid = self.warp_img_width / 2
            cv2.putText(self.edge, "neither", (20, 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)
        else:
            m = left_m if left_found else right_m
            mid = left_ini_x + self.lane_halfWidth if left_found else right_ini_x - self.lane_halfWidth
            state = "left only" if left_found else "right only"
            cv2.putText(self.edge, state, (20, 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)

        m_deg = math.degrees(math.tanh(-m)) # 왼쪽으로 기울면 m이 양수, 조향은 음수라 부호 반대 처리
        m_deg = 90 - m_deg if m_deg > 0 else -90 - m_deg        # 여기 이상함

        if -self.m_thr <= m_deg <= self.m_thr:
            ang = 0
        else:
            ang = m_deg

        self.angle_err = ang * self.deg_coeffi + (mid - 160) * self.dist_coeffi

        angle_str = "ang : " + str(round(m_deg, 2))
        pixel_str = "pixel_err : "+ str(round(mid-160, 2))
        err_str = "angle_err : " + str(round(self.angle_err, 2))
        cv2.putText(self.edge, angle_str, (20, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(self.edge, pixel_str, (20, 70), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(self.edge, err_str, (20, 90), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)

        cv2.imshow("src", self.frame)
        # cv2.imshow("warp", self.warp_img)
        cv2.imshow("edge", self.edge)

    def angle_pid(self):
        self.angle_d_err = self.angle_err - self.angle_p_err
        self.angle_p_err = self.angle_err
        self.angle_i_err += self.angle_err
        
        if self.angle_i_err > self.angle_max_i_err:
            self.angle_i_err = 0    # 적분 초기화
        
        self.angle_u = (self.angle_kp * self.angle_p_err) + (self.angle_ki * self.angle_i_err) +(self.angle_kd * self.angle_d_err)

    def pub_to_motor(self):
        self.angle_pid()
        self.steer_angle = int((self.angle_u - (-90)) * (50 - (-50)) / (90 - (-90)) + (-50))
            # (x-input_min)*(output_max-output_min)/(input_max-input_min)+output_min

        # motor_msg = xycar_motor()
        # motor_msg.speed = 4 #speed
        # motor_msg.angle = self.steer_angle
        # self.pub.publish(motor_msg) #둘다 int32

def main():
    rospy.init_node("lane_find", anonymous=True)

    lanefind = LaneFind()

    while not rospy.is_shutdown():
        if lanefind.frame.size != (640*480*3):
            continue    # 640*480 이미지 한 장이 모이기 전까지 대기
        
        lanefind.process_warp_img()
        lanefind.pub_to_motor()
        
        if cv2.waitKey(1) == 27:
            return

    rospy.spin()


if __name__=='__main__':
    main()


"""
추가할 사항

HSV 변환해서 적당한 값만 inrange로 변환
버즈아이뷰 없이 한 버전
차선 가운데로 가도록
"""