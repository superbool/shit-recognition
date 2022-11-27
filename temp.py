import time
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('./img/base.jpg')
# 行 列 通道
print(img.shape, img.size)
cv2.imshow('img', img)


# bgr通道
def show_bgr(img):
    b, g, r = cv2.split(img)
    cv2.imshow('b', b)
    cv2.imshow('g', g)
    cv2.imshow('r', r)


# hsv
def show_hsv(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.imshow('hsv', hsv)


def show_subtract(img):
    img2 = cv2.imread('./img/20221125082901.jpg')
    img3 = cv2.subtract(img2, img)
    cv2.imshow('img3', img3)
    gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray', gray)


def show_blur(img):
    # 均值滤波
    img_mean = cv2.blur(img, (5, 5))
    cv2.imshow('img_mean', img_mean)
    # 高斯滤波
    img_Guassian = cv2.GaussianBlur(img, (5, 5), 0)
    cv2.imshow('img_Guassian', img_Guassian)
    # 中值滤波
    img_median = cv2.medianBlur(img, 5)
    cv2.imshow('img_median', img_median)
    # 双边滤波
    img_bilater = cv2.bilateralFilter(img, 9, 75, 75)
    cv2.imshow('img_bilater', img_bilater)


#blur = cv2.GaussianBlur(img, (3, 3), 0)
#cv2.imshow('blur', blur)
# show_hsv(blur)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

while True:
    lower = np.array([0, 0, 80])
    upper = np.array([180, 30, 120])

    # set the lower and upper bounds for the green hue
    # lower = np.array([30, 0, 180])
    # upper = np.array([90, 10, 200])

    # create a mask for colour using inRange function
    mask = cv2.inRange(hsv, lower, upper)

    # perform bitwise and on the original image arrays using the mask
    res = cv2.bitwise_and(img, img, mask=mask)

    # display the images
    cv2.imshow("hsv", hsv)
    cv2.imshow("mask", mask)
    cv2.imshow("res", res)

    k = cv2.waitKey(33)
    if k == 27:  # Esc key to stop
        break

cv2.destroyAllWindows()

'''

color = ('b', 'g', 'r')
for i, col in enumerate(color):
    histr = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(histr, color=col)
    plt.xlim([0, 256])
plt.show()


gray = cv2.GaussianBlur(crop, (5, 5), 0)
edges = cv2.Canny(gray, 70, 210)
cv2.imshow("edged", edges)

'''
base = cv2.imread('base.jpeg')
base_gray = cv2.GaussianBlur(base, (5, 5), 0)
base_edges = cv2.Canny(base_gray, 70, 210)
cv2.imshow("base", base_edges)
imgInfo = crop.shape
dst = cv2.resize(base_edges, (imgInfo[1], imgInfo[0]))

res = cv2.bitwise_and(crop, crop, mask=dst)
cv2.imshow("res", res)
'''

contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(f"新轮廓数量：{len(contours)}")
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:4]

dest = []
for c in contours:
    area = cv2.contourArea(c)
    if area > 400:
        dest.append(c)
    print(str(area))

if dest:
    cv2.drawContours(crop, dest, -1, (0, 0, 255), 2)
    cv2.imshow('new', crop)

    cv2.drawContours(img, dest, -1, (0, 0, 255), -1, offset=(x + w, y + h))
    cv2.imshow('img', img)

    # x1, y1, w1, h1 = cv2.boundingRect(dest[0])
    # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
else:
    print("no shit")
'''
