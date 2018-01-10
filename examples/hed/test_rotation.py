import cv2
import imutils
import os


def test3():
    image_path = '../../data/test1.JPG'
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    for angle in range(0, 380, 20):
        rotated_image = imutils.rotate_bound(image, angle)
        filename = `angle` + '.png'
        save_path = os.path.join('../../data/test_rotation', filename)
        cv2.imwrite(save_path, rotated_image)


test3()