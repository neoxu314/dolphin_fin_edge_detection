import numpy as np
from PIL import Image
import cv2


image = np.array(Image.open('../../data/test.png'))
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        # print(image[i][j])
        if not (image[i][j][0] == 255 and image[i][j][1] == 255 and image[i][j][2] == 255):
            # print('yes')
            image[i][j][0] = 0
            image[i][j][1] = 0
            image[i][j][2] = 0

cv2.imshow('img', image)
cv2.waitKey(0)