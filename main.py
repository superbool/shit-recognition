import cv2
import time

import numpy as np

# 相对路径下读取图片
img = cv2.imread('./img/20221125082713.jpg')
cv2.imshow('img', img)

# blur = cv2.GaussianBlur(img, (5, 5), 0)
# cv2.imshow('img1', blur)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# set the lower and upper bounds for the green hue
lower = np.array([85, 10, 0])
upper = np.array([120, 90, 255])

# create a mask for colour using inRange function
mask = cv2.inRange(hsv, lower, upper)

# perform bitwise and on the original image arrays using the mask
res = cv2.bitwise_and(img, img, mask=mask)

# 膨胀待定
# kernel = np.ones((10, 10), dtype=np.uint8)
# dilate = cv2.dilate(mask, kernel, 1)  # 1:迭代次数，也就是执行几次膨胀操作
# cv2.imshow("dilate", dilate)

# display the images
cv2.imshow("hsv", hsv)
cv2.imshow("mask", mask)
cv2.imshow("res", res)

# 查找轮廓
contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# print(f"轮廓数量：{len(contours)}")
# 按面积排序
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]
contour = contours[0]

new_img = img.copy()

# 轮廓
cv2.drawContours(new_img, contours, -1, (0, 0, 255), 2)
# cv2.imshow("contours", new_img)

# 外接矩形
x, y, w, h = cv2.boundingRect(contour)
print(x, y, w, h)
cv2.rectangle(new_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.imshow("contour", new_img)

# 裁剪
crop = img[y:y + h, x:x + w]
cv2.imshow('crop', crop)
cv2.imwrite('./img/base.jpg', crop)

# TODO 判断是否是厕所


while (1):
    k = cv2.waitKey(33)
    if k == 27:  # Esc key to stop
        break

cv2.destroyAllWindows()
