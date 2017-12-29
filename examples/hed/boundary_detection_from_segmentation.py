'''
Gets the extracted boundary from the segmentation image by using morphological method
'''
import cv2
import sys
import os
import re
import numpy as np
from PIL import Image


def get_image_paths(dir_path):
    '''
    Returns a list of paths to JPG and PNG files in dir_path.
    :param dir_path: the path of the input directory
    :return: a list of paths to JPG and PNG files in dir_path.
    '''
    regex = re.compile(r'.*\.(jpe?g|png)$', re.IGNORECASE)
    paths = [os.path.join(dir_path, f) for f in os.listdir(dir_path)
             if os.path.isfile(os.path.join(dir_path, f)) and regex.match(f)]
    return paths


def get_boundary_overlay(input_dir_path, boundary_save_path, overlay_save_path):
    '''
    Saves the overlay images and images of extracted boundary to the target path. The overlay image is the overlay the
    extracted boundary on the original input image.
    :param input_dir_path: The path to input images
    :param boundary_save_path: The path to save the images of extracted boundary
    :param overlay_save_path: The path to save the overlay images
    :return: None
    '''
    input_image_paths = get_image_paths(input_dir_path)

    for input_path in input_image_paths:
        print('********Processing boundary: ', input_path)

        # Converts the non-transparent pixel to black
        original_image = np.array(Image.open(input_path))
        image = np.array(Image.open(input_path))

        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image[i][j][3] > 0:
                    image[i][j][0] = 0
                    image[i][j][1] = 0
                    image[i][j][2] = 0


        # constructs a 3 x 3 element
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilate = cv2.dilate(image, element)
        erode = cv2.erode(image, element)

        # subtracts eroded image from dilated image to get the boundary
        result = cv2.absdiff(dilate, erode)

        # binarise grayscale image
        retval, result = cv2.threshold(result, 40, 255, cv2.THRESH_BINARY)

        # reverses colour
        # result = cv2.bitwise_not(result)

        result_black = result.copy()

        for i in range(result_black.shape[0]):
            for j in range(result_black.shape[1]):
                if result_black[i][j][3] == 0:
                    result_black[i][j][3] = 255
                    result_black[i][j][0] = 0
                    result_black[i][j][1] = 0
                    result_black[i][j][2] = 0

        path, filename = os.path.split(input_path)
        output_path = os.path.join(boundary_save_path, filename)
        cv2.imwrite(output_path, result_black)

        get_overlay_image(original_image, result, overlay_save_path, filename)


def get_overlay_image(image1, image2, overlay_save_path, filename):
    '''
    Save the overlay image.
    :param image1: The images of extracted boundary
    :param image2: The original input images
    :param overlay_save_path: The directory path to the save the overlay images
    :param filename: The filename of the original input images
    :return: None
    '''
    alpha = 0.2
    image2 = cv2.bitwise_not(image2)
    cv2.addWeighted(image2, alpha, image1, 1, 0, image1)
    # cv2.imshow('overlay', image1)
    # cv2.waitKey(0)
    output_path = os.path.join(overlay_save_path, filename)
    print('********Processing overlay: ', output_path)
    cv2.imwrite(output_path, image1)


def main(argv):
    # Gets args
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-image-path', metavar='Path',
                        help='Path to the input segmentation image (default: None)')
    parser.add_argument('--boundary-save-path', metavar='Path', help='Path to save the boundary image (default: None)')
    parser.add_argument('--overlay-save-path', metavar='Path', help='Path to save the overlay image (default: None)')
    args = parser.parse_args(argv)
    input_path =args.input_image_path
    boundary_save_path = args.boundary_save_path
    overlay_save_path = args.overlay_save_path

    # Gets the image of extracted boundary
    get_boundary_overlay(input_path, boundary_save_path, overlay_save_path)


if __name__ == '__main__':
    main(sys.argv[1:])
