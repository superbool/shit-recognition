import numpy as np
import cv2 as cv

image = cv.imread('./img/20221125082713.jpg')


def sI(val):
    cv.imshow('Window', image)


cv.namedWindow('Window', cv.WINDOW_AUTOSIZE)

cv.imshow("Window", image)

cv.createTrackbar('Trackbar', 'Window', 1, 100, sI)

while True:
    ch = cv.waitKey(0)
    if ch == 27:
        break

cv.destroyAllWindows()


def on_trackbar(val):
    print(val)

def createTrackbar():
    cv2.namedWindow("HSV")
    cv2.resizeWindow("HSV", 600, 400)

    cv2.createTrackbar("SENS", "HSV", 0, 179, on_trackbar)  # 灵敏度
    cv2.createTrackbar("HUE", "HSV", 0, 179, on_trackbar)  # 色相

    cv2.createTrackbar("SAT_Min", "HSV", 0, 255, on_trackbar)  # 饱和度下限
    cv2.createTrackbar("SAT_Max", "HSV", 255, 255, on_trackbar)  # 饱和度上限

    cv2.createTrackbar("VAL_Min", "HSV", 0, 255, on_trackbar)  # 明度下限
    cv2.createTrackbar("VAL_Max", "HSV", 255, 255, on_trackbar)  # 明度上限

def hue_range(val, sensitivity):
    up = val + sensitivity
    down = val - sensitivity
    return 0 if down < 0 else down, 179 if up > 179 else up


def getBoundary():
    sens = cv2.getTrackbarPos("SENS", "HSV")
    hue = cv2.getTrackbarPos("HUE", "HSV")

    h_min, h_max = hue_range(hue, sens)

    s_min = cv2.getTrackbarPos("SAT_Min", "HSV")
    s_max = cv2.getTrackbarPos("SAT_Max", "HSV")

    v_min = cv2.getTrackbarPos("VAL_Min", "HSV")
    v_max = cv2.getTrackbarPos("VAL_Max", "HSV")

    print(','.join([str(x) for x in (h_min, s_min, v_min, h_max, s_max, v_max)]))

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])

    return lower, upper