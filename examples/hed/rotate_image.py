# import the necessary packages
import cv2
import numpy as np

def rotateImage(image, angle):
  image_center = tuple(np.array(image.shape)/2)
  rot_mat = cv2.getRotationMatrix2D(image_center,angle,1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape,flags=cv2.INTER_LINEAR)
  return result


# load the image and show it
image1 = cv2.imread("../../data/test.png", cv2.IMREAD_UNCHANGED)

# cv2.imshow("original", image)
# cv2.waitKey(0)

# print(image.shape)


# grab the dimensions of the image and calculate the center of the image
(h, w) = image1.shape[:2]
center = (w / 2, h / 2)

# rotate the image by 180 degrees
M = cv2.getRotationMatrix2D(center, 40, 1.0)
rotated = cv2.warpAffine(image1, M, (w, h), flags=cv2.INTER_LINEAR)
# cv2.imshow("rotated", rotated)
# cv2.waitKey(0)
#
# for i in range(rotated.shape[0]):
#     for j in range(rotated.shape[1]):
#         if rotated[i][j][0] == 0 and rotated[i][j][1] == 0 and rotated[i][j][2] == 0:
#             rotated[i][j][0] = 255
#             rotated[i][j][1] = 255
#             rotated[i][j][2] = 255




cv2.imwrite('../../data/test_rotate.png', rotated)

# rotated = rotateImage(image1, 40)
# cv2.imshow("rotated", rotated)
# cv2.waitKey(0)

