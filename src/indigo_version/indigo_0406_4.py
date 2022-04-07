#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2, random, math, copy
import rospy, rospkg
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from xycar_msgs.msg import xycar_motor
import sys
import os
import signal

class PID():
    def __init__(self, kp=0.45, ki=0.0007, kd=0.15):
        self.Kp = kp
        self.Ki = ki
        self.Kd = kd
        self.p_error = 0.0
        self.i_error = 0.0
        self.d_error = 0.0

    def set(self,kp,ki,kd):
        self.Kp = kp
        self.Ki = ki
        self.Kd = kd
        self.p_error = 0.0
        self.i_error = 0.0
        self.d_error = 0.0

    def pid_control(self, cte):
        self.d_error = cte - self.p_error
        self.p_error = cte
        self.i_error += cte
        if abs(self.i_error) > 100000:
            self.i_error = 0

        return self.Kp * self.p_error + self.Ki * self.i_error + self.Kd * self.d_error

class MAF:

    def __init__(self,n):
        self.n = n
        self.data = [1] * self.n
        self.weights = list(range(1,self.n+1))

    def add_data(self, new_data):
        if len(self.data) < self.n:
            self.data.append(new_data)
        else:
            self.data = self.data[1:] + [new_data]

    def get_data(self):
        return float(sum(self.data))/len(self.data)

    def get_w_data(self):
        s = 0
        for i, x in enumerate(self.data):
            s += x * self.weights[i]
        return float(s) / sum(self.weights[:len(self.data)])

class Houghline_Detect:

    def signal_handler(self, sig, frame):
        os.system('killall -9 python rosout')
        sys.exit(0)

    def __init__(self):

        signal.signal(signal.SIGINT, self.signal_handler)

        self.m_a_f = MAF(20)
        self.pid=PID()

        self.image = np.empty(shape=[0])
        self.bridge = CvBridge()
        self.pub = None
        self.Width = 640
        self.Height = 480
        self.Offset = 380
        self.Gap = 40
        self.pid_p = 0.45
        self.pid_i = 0.0007
        self.pid_d = 0.015
        self.angle = 0.0
        self.count = 0
        self.trackbar()
        
        self.prev_rpos = 0
        self.prev_lpos = 0

    def nothing(self, x):
        pass

    def change_offset(self, x):
        self.Offset = x

    def change_P(self, x):
        self.pid_p = float(x / 100)
        self.pid.set(self.pid_p,self.pid_i,self.pid_d)
    def change_I(self, x):
        self.pid_i = float(x / 10000)
        self.pid.set(self.pid_p, self.pid_i, self.pid_d)
    def change_D(self, x):
        self.pid_d = float(x / 1000)
        self.pid.set(self.pid_p, self.pid_i, self.pid_d)

    def trackbar(self):
        cv2.namedWindow("pallete")
        cv2.createTrackbar('Offset', "pallete", 380, 400, self.change_offset)
        cv2.createTrackbar('PID_P', "pallete", 45, 100, self.change_P)
        cv2.createTrackbar('PID_I', "pallete", 7, 10, self.change_I)
        cv2.createTrackbar('PID_D', "pallete", 15, 100, self.change_D)
        cv2.createTrackbar('speed', "pallete", 0, 30, self.nothing)
        cv2.createTrackbar('middle_thresh', "pallete", 10, 90, self.nothing)

    def img_callback(self, data):
        self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")

    # publish xycar_motor msg
    def drive(self, Angle, Speed):
        msg = xycar_motor()
        if Angle > 50:
            Angle = 50
        if Angle < -50:
            Angle = -50
        msg.angle = int(Angle)

        msg.speed = int(Speed - (0.2 * abs(Angle)))
        
        print("Angle = ", Angle, " / Speed = ", msg.speed)
        self.pub.publish(msg)

    # specific case
    def stop_back(self):
        print("--------------stop---------------")
        rate = rospy.Rate(10)

        for i in range(5):
            self.drive(-self.angle, -10)
            #self.drive(0,-10)
            rate.sleep()
        
    # draw lines
    def draw_lines(self, img, lines):
        for line in lines:
            x1, y1, x2, y2 = line[0]
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            img = cv2.line(img, (x1, y1 + self.Offset), (x2, y2 + self.Offset), color, 2)
        return img

    # draw rectangle
    def draw_rectangle(self, img, lpos, rpos, offset=0):
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
        cv2.rectangle(img, (self.Width / 2 - 5, 15 + offset),
                      (self.Width / 2 + 5, 25 + offset),
                      (0, 0, 255), 2)
        return img

    # left lines, right lines
    def divide_left_right(self, lines):
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

        middle_thresh = cv2.getTrackbarPos('middle_thresh', "pallete")


        for j in range(len(slopes)):
            Line = new_lines[j]
            slope = slopes[j]

            x1, y1, x2, y2 = Line

            if (slope < 0) and (x2 < self.Width / 2 - middle_thresh):
                left_lines.append([Line.tolist()])
            elif (slope > 0) and (x1 > self.Width / 2 + middle_thresh):
                right_lines.append([Line.tolist()])

        return left_lines, right_lines

    # get average m, b of lines
    def get_line_params(self, lines):
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

    # get lpos, rpos
    def get_line_pos(self, img, lines, left=False, right=False):

        m, b = self.get_line_params(lines)

        if m == 0 and b == 0:
            if left:
                pos = -1
            if right:
                pos = self.Width + 1
        else:
            y = self.Gap / 2
            pos = (y - b) / m

            b += self.Offset
            x1 = (self.Height - b) / float(m)
            x2 = ((self.Height / 2) - b) / float(m)
            
            #draw line
            #cv2.line(img, (int(x1), self.Height), (int(x2), (self.Height / 2)), (255, 0, 0), 3)

        return img, int(pos)

    # show image and return lpos, rpos
    def process_image(self, frame):

        # gray
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #normalize
        #type_flag는 alpha와 beta 구간으로 정규화하는 cv2.NORM_MINMAX
        #전체 합으로 나누는 cv2.NORM_L1 
        #단위 벡터로 정규화하는 cv2.NORM_L2 
        #최댓값으로 나누는 cv2.NORM_INF
	norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        # blur
        kernel_size = 5
        blur_gray = cv2.GaussianBlur(norm, (kernel_size, kernel_size), 0)

        # canny edge
        edge_img = cv2.Canny(np.uint8(blur_gray), 140, 70)

        # HoughLinesP roi 
        roi = edge_img[self.Offset: self.Offset + self.Gap, 0: self.Width]

        # cv2.HoughLinesP()
        all_lines = cv2.HoughLinesP(roi, 1, math.pi / 180, 30, 30, 10)

        # divide left, right lines
        if all_lines is None:
            return 0, 640
        left_lines, right_lines = self.divide_left_right(all_lines)

        # get center of lines
        frame, lpos = self.get_line_pos(frame, left_lines, left=True)
        frame, rpos = self.get_line_pos(frame, right_lines, right=True)

        # draw lines
        #frame = self.draw_lines(frame, left_lines)
        #frame = self.draw_lines(frame, right_lines)
        #frame = cv2.line(frame, (230, 235), (410, 235), (255, 255, 255), 2)

        # draw rectangle
        #frame = self.draw_rectangle(frame, lpos, rpos, offset=self.Offset)
        # roi2 = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        # roi2 = draw_rectangle(roi2, lpos, rpos)

        # show image
        #cv2.imshow('calibration', frame)
        return lpos, rpos

    def check_prev_pos(self,rpos,lpos):


    def start(self):

        rospy.init_node('auto_drive')
        self.pub = rospy.Publisher("xycar_motor", xycar_motor, queue_size=1)
        rospy.Subscriber("/usb_cam/image_raw/", Image, self.img_callback)

        print("---------- Indigo Start! ----------")
        rospy.sleep(2)

        r_l_gap = 500

        while True:
            while not self.image.size == (640 * 480 * 3):
                continue

            lpos, rpos = self.process_image(self.image)
            if lpos == -1 and rpos != self.Width + 1:
                lpos = rpos - r_l_gap 

            elif rpos == self.Width + 1 and lpos != -1:
                rpos = lpos + r_l_gap 

            elif rpos == self.Width + 1 and lpos == -1:
                self.count += 1  
                if self.count >= 3:
                    self.stop_back()
                continue
                    
            else:
                if rpos - lpos < r_l_gap:
                    if abs(rpos-self.prev_rpos) < 100:
                        lpos = rpos - r_l_gap 
                    if abs(lpos-self.prev_lpos) < 100:
                        rpos = lpos - r_l_gap 

            
            self.count = 0

            center = (lpos + rpos) / 2
            error = (center - self.Width / 2 + 10)
            
            #angle modify 
            self.m_a_f.add_data(error)
            avg_angle = self.m_a_f.get_data()
            self.angle = self.pid.pid_control(avg_angle)

            speed = cv2.getTrackbarPos('speed', "pallete")

            self.drive(self.angle , speed)
            self.prev_rpos = rpos
            self.prev_lpos = lpos

            if cv2.waitKey(1) & 0xFF == ord('q'):
                if cv2.waitKey() & 0xFF == ord('q'):
                    break

if __name__ == '__main__':
    a = Houghline_Detect()
    a.start()
