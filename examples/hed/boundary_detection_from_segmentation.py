'''
Gets the extracted boundary from the segmentation image by using morphological method
'''
import cv2
import sys
import os
import re


def get_image_paths(dir_path):
    """ Returns a list of paths to JPG and PNG files in dir_path. """
    regex = re.compile(r'.*\.(jpe?g|png)$', re.IGNORECASE)
    paths = [os.path.join(dir_path, f) for f in os.listdir(dir_path)
             if os.path.isfile(os.path.join(dir_path, f)) and regex.match(f)]
    return paths


def get_boundary(input_dir_path, output_dir_path):
    '''
    Gets the image of extracted boundary.

    :param input_dir_path: The path to input images
    :param output_dir_path: The path to output images
    :return: None
    '''
    input_image_paths = get_image_paths(input_dir_path)

    for input_path in input_image_paths:
        print('********Processing image: ', input_path)

        image = cv2.imread(input_path, 0)
        # constructs a 3 x 3 element
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilate = cv2.dilate(image, element)
        erode = cv2.erode(image, element)

        # subtracts eroded image from dilated image to get the boundary
        result = cv2.absdiff(dilate, erode)

        # binarise grayscale image
        retval, result = cv2.threshold(result, 40, 255, cv2.THRESH_BINARY);

        # reverses colour
        # result = cv2.bitwise_not(result)

        # shows images
        # cv2.imshow("result", result)

        path, filename = os.path.split(input_path)
        output_path = os.path.join(output_dir_path, filename)
        cv2.imwrite(output_path, result)


def main(argv):
    # Gets args
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-image-path', metavar='Path',
                        help='Path to the input segmentation image (default: None)')
    parser.add_argument('--output-image-path', metavar='Path', help='Path to the ouput image (default: None)')
    args = parser.parse_args(argv)
    input_path =args.input_image_path
    output_path =args.output_image_path

    # Gets the image of extracted boundary
    get_boundary(input_path, output_path)


if __name__ == '__main__':
    main(sys.argv[1:])
