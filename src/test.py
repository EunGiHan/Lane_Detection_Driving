#-*- coding:utf-8 -*-

import cv2
import numpy as np
from cv_bridge import CvBridge


########일반배열 원근 변환
lx = np.arange(90, 101, 1).reshape(1, 11)
ly = np.arange(90, 101, 1).reshape(1, 11)
mat = np.transpose(np.vstack([lx, ly]))

src = np.zeros((480, 640)).astype(np.uint8)
for m in mat:
    src[m[0], m[1]] = 255
cv2.imshow("warp", src)

# warp_img_width = 320    # 와핑 가로
# warp_img_height = 240   # 와핑 세로

# warp_x_margin = 20  # 마진은 왜 주는 거야....???
# warp_y_margin = 3
# warp_src = np.array([
#             [230 - warp_x_margin, 300 - warp_y_margin],
#             [45 - warp_x_margin, 450 + warp_y_margin],
#             [445 + warp_x_margin, 300 - warp_y_margin],
#             [610 + warp_x_margin, 450 + warp_y_margin]
#         ], dtype = np.float32)  # 위 왼쪽 / 아래 왼쪽 / 위 오른쪽 / 아래 오른쪽

# warp_dst = np.array([
#     [0, 0],
#     [0, warp_img_height],
#     [warp_img_width, 0],
#     [warp_img_width, warp_img_height]
# ], dtype = np.float32)

# perspec_mat_inv = cv2.getPerspectiveTransform(warp_dst, warp_src)
# src_warp = cv2.warpPerspective(mat, perspec_mat_inv, (2, 11))

# cv2.imshow("warp", mat)
# cv2.imshow("frame", src_warp)

cv2.waitKey()

########### 화면 밖 직선
src = np.zeros((480, 640)).astype(np.uint8)
src = cv2.line(src, (-1.1635465341, -1.5646854), (100, 100), 255, 2)
cv2.imshow("src", src)
cv2.waitKey()