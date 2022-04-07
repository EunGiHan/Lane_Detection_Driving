#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from xycar_msgs.msg import xycar_motor

msg = xycar_motor()
msg.angle = 50
msg.speed = 20

rate = rospy.Rate(10)
rospy.init_node('auto_drive')
pub = rospy.Publisher("xycar_motor", xycar_motor, queue_size=1)

for _ in range(5):
    pub.publish(msg)
    rate.sleep()

for _ in range(5):
    msg.angle = -50
    pub.publish(msg)
    rate.sleep()