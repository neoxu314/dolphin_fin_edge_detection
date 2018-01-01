'''
Data preprocessing for fin images
'''

import cv2
import sys
import os
import re
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from PIL import Image
import numpy as np
import Augmentor


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

        # get_overlay_image(original_image, result, overlay_save_path, filename)


def get_overlay_image(image1, image2, overlay_save_path, filename):
    '''
    Save the overlay image.
    :param image1: The images of extracted boundary
    :param image2: The original input images
    :param overlay_save_path: The directory path to the save the overlay images
    :param filename: The filename of the original input images
    :return: None
    '''
    alpha = 0.4
    image2 = cv2.bitwise_not(image2)
    cv2.addWeighted(image2, alpha, image1, 1, 0, image1)
    # cv2.imshow('overlay', image1)
    # cv2.waitKey(0)
    output_path = os.path.join(overlay_save_path, filename)
    print('********Processing overlay: ', output_path)
    cv2.imwrite(output_path, image1)


def get_ground_truth(directories):
    for directory in directories:
        input_dir_path = os.path.join(directory, 'augmentation')

        boundary_save_path = os.path.join(input_dir_path, 'gt_boundary')
        if not os.path.exists(boundary_save_path):
            os.makedirs(boundary_save_path)

        overlay_save_path = os.path.join(input_dir_path, 'gt_overlay')
        if not os.path.exists(overlay_save_path):
            os.makedirs(overlay_save_path)

        get_boundary_overlay(input_dir_path, boundary_save_path, overlay_save_path)


def begin_with_dot(path):
    split_path = path.split('/')
    if split_path[-1].startswith('._'):
        # print(split_path[-1])
        return True
    return False


def get_image_paths(dir_path):
    """ Returns a list of paths to JPG and PNG files in dir_path. """
    regex = re.compile(r'.*\.(jpe?g|png)$', re.IGNORECASE)
    paths = [os.path.join(dir_path, f) for f in os.listdir(dir_path)
             if os.path.isfile(os.path.join(dir_path, f)) and regex.match(f)]

    new_paths = [x for x in paths if not begin_with_dot(x)]

    return new_paths


def get_list_from_multilevelDir(input_image_path):
    # root_directory = '../../data/Final_Database'
    root_directory = input_image_path

    folder_names = []
    sub_directories = []
    image_paths = []

    for root, dirs, files in os.walk(root_directory):
        if root == root_directory:
            folder_names = dirs
        # print('****root****: ', root)
        # print('****dirs****: ', dirs)
        # print('****files****: ', files)

    print(folder_names)
    print(len(folder_names))
    for folder_name in folder_names:
        path = os.path.join(root_directory, folder_name)
        # path = os.path.join(path, '/PNG')
        path = path + '/PNG'
        sub_directories.append(path)

    print(sub_directories)

    for directory in sub_directories:
        image_paths.extend(get_image_paths(directory))

    print('*********************image paths**********************')
    print(image_paths)

    return sub_directories, image_paths


def output_lst_file_from_list(image_paths, output_image_lst_file_path):
    # list_path = '../../data/segmentation_fin.lst'
    list_path = output_image_lst_file_path
    list_file = open(list_path, 'w')
    for path in image_paths:
        npath = path[path.find('Final_Database'):]
        list_file.write("%s\n" % npath)


def data_augmentation_using_Keras(directories):
    rotation_range = 30
    zoom_range = 0.5

    datagen = ImageDataGenerator(
        rotation_range=rotation_range)

    for directory in directories:
        images_list = get_image_paths(directory)


        augmented_image_save_dir = os.path.join(directory, 'augmentation')
        print('save augmented image: ', augmented_image_save_dir)
        if not os.path.exists(augmented_image_save_dir):
            os.makedirs(augmented_image_save_dir)

        for image in images_list:
            img = load_img(image)  # this is a PIL image, please replace to your own file path
            x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
            x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

            # the .flow() command below generates batches of randomly transformed images
            # and saves the results to the `preview/` directory

            i = 0
            for batch in datagen.flow(x,
                                      batch_size=1,
                                      save_to_dir=augmented_image_save_dir,
                                      save_prefix='test',
                                      save_format='png'):
                i += 1

                if i > 20:
                    break  # otherwise the generator would loop indefinitely


def data_augmentation_using_Augmentor(directories):
    for directory in directories:
        p = Augmentor.Pipeline(directory)
        p.rotate(probability=1, max_left_rotation=10, max_right_rotation=10)
        p.sample(10)


def data_augmentation(directories):
    for directory in directories:
        images_list = get_image_paths(directory)

        augmented_image_save_dir = os.path.join(directory, 'augmentation')
        print('save augmented image: ', augmented_image_save_dir)
        if not os.path.exists(augmented_image_save_dir):
            os.makedirs(augmented_image_save_dir)

        for image in images_list:
            for rotation_angle in range(0, 40, 10):
                # load the image and show it
                img = cv2.imread(image, cv2.IMREAD_UNCHANGED)

                # grab the dimensions of the image and calculate the center of the image
                (h, w) = img.shape[:2]
                center = (w / 2, h / 2)

                # rotate the image by 180 degrees
                M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
                rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)

                # resize
                dim_0_5 = (int(0.5 * w), int(0.5 * h))
                resized_0_5 = cv2.resize(rotated, dim_0_5, interpolation=cv2.INTER_AREA)
                dim_1_5 = (int(1.5 * rotated.shape[1]), int(rotated.shape[0] * 1.5))
                resized_1_5 = cv2.resize(rotated, dim_1_5, interpolation=cv2.INTER_AREA)

                path, filename = os.path.split(image)
                filename = os.path.splitext(filename)[0]


                rotated_image_name = filename + '_rotation' + `rotation_angle` + '.png'
                rotated_image_name_0_5 = filename + '_rotation' + `rotation_angle` + '_zoom0_5' + '.png'
                rotated_image_name_1_5 = filename + '_rotation' + `rotation_angle` + '_zoom1_5' + '.png'

                print('**************saving original: ', rotated_image_name)
                print('**************saving zoom 0.5: ', rotated_image_name_0_5)
                print('**************saving zoom 1.5: ', rotated_image_name_1_5)

                rotated_image_save_path = os.path.join(augmented_image_save_dir, rotated_image_name)
                rotated_image_save_path_0_5 = os.path.join(augmented_image_save_dir, rotated_image_name_0_5)
                rotated_image_save_path_1_5 = os.path.join(augmented_image_save_dir, rotated_image_name_1_5)

                cv2.imwrite(rotated_image_save_path, rotated)
                cv2.imwrite(rotated_image_save_path_0_5, resized_0_5)
                cv2.imwrite(rotated_image_save_path_1_5, resized_1_5)

            for rotation_angle in range(-10, -40, -10):
                # load the image and show it
                img = cv2.imread(image, cv2.IMREAD_UNCHANGED)

                # grab the dimensions of the image and calculate the center of the image
                (h, w) = img.shape[:2]
                center = (w / 2, h / 2)

                # rotate the image by 180 degrees
                M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
                rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)

                # resize
                dim_0_5 = (int(0.5 * w), int(0.5 * h))
                resized_0_5 = cv2.resize(rotated, dim_0_5, interpolation=cv2.INTER_AREA)
                dim_1_5 = (int(1.5 * rotated.shape[1]), int(rotated.shape[0] * 1.5))
                resized_1_5 = cv2.resize(rotated, dim_1_5, interpolation=cv2.INTER_AREA)

                path, filename = os.path.split(image)
                filename = os.path.splitext(filename)[0]


                rotated_image_name = filename + '_rotation' + `rotation_angle` + '.png'
                rotated_image_name_0_5 = filename + '_rotation' + `rotation_angle` + '_zoom0_5' + '.png'
                rotated_image_name_1_5 = filename + '_rotation' + `rotation_angle` + '_zoom1_5' + '.png'

                print('**************saving original: ', rotated_image_name)
                print('**************saving zoom 0.5: ', rotated_image_name_0_5)
                print('**************saving zoom 1.5: ', rotated_image_name_1_5)

                rotated_image_save_path = os.path.join(augmented_image_save_dir, rotated_image_name)
                rotated_image_save_path_0_5 = os.path.join(augmented_image_save_dir, rotated_image_name_0_5)
                rotated_image_save_path_1_5 = os.path.join(augmented_image_save_dir, rotated_image_name_1_5)

                cv2.imwrite(rotated_image_save_path, rotated)
                cv2.imwrite(rotated_image_save_path_0_5, resized_0_5)
                cv2.imwrite(rotated_image_save_path_1_5, resized_1_5)


def parse_args(argv):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('cmd', choices=['get_lst_file_from_dir', 'get_trainpair_lst_file', 'data_augmentation', 'get_ground_truth'],
                        help='The command to run')
    parser.add_argument('--input-image-path', metavar='Path',
                        help='The path to the directory of input images (default: None)')
    parser.add_argument('--input-ground-truth-path', metavar='Path',
                        help='The path to the directory of input ground truth (default: None)')
    parser.add_argument('--output-image-lst-file-path', metavar='Path',
                        help='The path to save the output lst file of images in one directory (default: None)')
    parser.add_argument('--output-trainpair-lst-file-path', metavar='Path',
                        help='The path to save the output lst file of train pair '
                             '(original images and corresponding ground truth) (default: None)')
    parser.add_argument('--output-augmented-images-path', metavar='Path',
                        help='The path to save the output augmented images')
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    cmd = args.cmd
    input_image_path = args.input_image_path
    input_ground_truth_path = args.input_ground_truth_path
    output_image_lst_file_path = args.output_image_lst_file_path
    output_trainpair_lst_file_path = args.output_trainpair_lst_file_path
    output_augmented_images_path = args.output_augmented_images_path

    if cmd == 'get_lst_file_from_dir':
        # Outputs a lst file which contains the paths of every images in the directory input_image_path
        # to output_image_lst_file_path
        sub_directories, image_paths = get_list_from_multilevelDir(input_image_path)
        output_lst_file_from_list(image_paths, output_image_lst_file_path)
    elif cmd == 'get_trainpair_lst_file':
        print('get_trainpair_lst_file')
    elif cmd == 'data_augmentation':
        print('data_augmentation')
        sub_directories, image_paths = get_list_from_multilevelDir(input_image_path)
        data_augmentation(sub_directories)
        # data_augmentation_using_Keras(sub_directories)
        # data_augmentation_using_Augmentor(sub_directories)
    elif cmd == 'get_ground_truth':
        sub_directories, image_paths = get_list_from_multilevelDir(input_image_path)
        get_ground_truth(sub_directories)


if __name__ == '__main__':
    main(sys.argv[1:])