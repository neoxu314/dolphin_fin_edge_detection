from PIL import Image
import numpy as np
import cv2

img = np.array(Image.open('../../data/test_boundary_input/1.png'))

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if img[i][j][3] > 0:
            img[i][j][0] = 0
            img[i][j][1] = 0
            img[i][j][2] = 0
        # elif img[i][j][3] == 0:
        #     img[i][j][3] = 255

cv2.imshow('img', img)
cv2.waitKey(0)
