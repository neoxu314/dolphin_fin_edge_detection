import cv2
import numpy as np

blank_image = np.zeros((554,759,3), np.uint8)
blank_image = cv2.bitwise_not(blank_image)
img = cv2.imread('../../data/test_boundary_output/1.png', cv2.IMREAD_GRAYSCALE)
img = cv2.bitwise_not(img)
cv2.imshow('img', img)
# cv2.waitKey(0)

# Set up the SimpleBlobdetector with default parameters.
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 0
params.maxThreshold = 100000

# Filter by Area.
params.filterByArea = False
# params.minArea = 0

# Filter by Circularity
params.filterByCircularity = False
# params.minCircularity = 2

# Filter by Convexity
params.filterByConvexity = False
# params.minConvexity = 0.5

# Filter by Inertia
params.filterByInertia = False
# params.minInertiaRatio = 0.5

blob_detector = cv2.SimpleBlobDetector_create(params)


keypoints = blob_detector.detect(img)
im_with_keypoints = cv2.drawKeypoints(blank_image, keypoints, np.array([]), (0,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('keypoints', im_with_keypoints)
cv2.waitKey(0)

# for i in keypoints:
#     print(type(i))