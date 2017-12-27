import cv2
import numpy as np

img = cv2.imread('../../data/test_boundary_output/1.png')
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
(_, contours, _) = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# biggest area
target = max(contours, key=lambda x: cv2.contourArea(x))
cv2.drawContours(img, [target], -1, [0, 0, 255], -1) # debug
# just example of fitting
x = target[:, :, 0].flatten()
y = target[:, :, 1].flatten()
poly = np.poly1d(np.polyfit(x, y, 5))
for _x in range(min(x), max(x), 5): # too lazy for line/curve :)
    cv2.circle(img, (_x, int(poly(_x))), 3, [0, 255, 0])

cv2.imshow('result', img)
cv2.waitKey(0)