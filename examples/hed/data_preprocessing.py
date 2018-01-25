'''
Data preprocessing for fin images
'''

import cv2
import sys
import os
import re
import numpy as np
import math
from shutil import copyfile
import scipy.io
import skimage.io
import skimage.transform
from numpy.linalg import inv
import matplotlib.pyplot as plt
# import imutils


def test(images_path):
    db_path = '../../data/new_dataset/out.mat'
    orig_img_path = '../../data/new_dataset/big/'
    edge_img_path = '../../data/new_dataset/e_edge/'
    right_image_path = '../../data/new_dataset/e_edge/'
    data_augmentation(db_path, orig_img_path, edge_img_path, right_image_path)

    # convert_background_to_transparency('../../data/new_dataset/e_edge/')

    # get_train_pair_list('../../data/new_dataset2/big/augmentation/', '../../data/new_dataset2/e_edge/augmentation/', '../../data/new_dataset2/train_pair.lst')
    # get_square_test_set()
    # get_train_pair_list('../../data/test1/square/big/', '../../data/test1/square/e_edge/', '../../data/test1/train_pair.lst')
    # get_train_pair_list('../../data/test1/rectangle/big/', '../../data/test1/rectangle/e_edge/', '../../data/test1/train_pair.lst')
    # resize_width(images_path)
    # resize_width('../../data/new_dataset/e_edge/', 1000)
    # find_the_biggest_width('../../data/new_dataset/e_edge/augmentation/')
    # get_rectangle_test_set()
    # gradient_resize('../../data/test1/square/big/')
    # gradient_resize('../../data/test1/rectangle/big/')
    # show_size_of_images('../../data/test1/square/big/resized/')
    # show_size_of_images('../../data/test1/rectangle/big/resized/')
    # gradient_resize(images_path)
    # choose_n_images(1000)


def choose_n_images(n):
    # orig_images_path = '../../data/new_dataset/big/augmentation'
    orig_images_path = '/media/neo/92cd53f3-fc9f-4db8-b68e-faf606561e34/home/neo/PycharmProjects/hed_origin/data/new_dataset/big/augmentation/'
    # edge_images_path = '../../data/new_dataset/e_edge/augmentation'
    edge_images_path = '/media/neo/92cd53f3-fc9f-4db8-b68e-faf606561e34/home/neo/PycharmProjects/hed_origin/data/new_dataset/e_edge/augmentation/'

    # orig_image_dst_folder_path = '../../data/new_dataset1/big/augmentation/'
    orig_image_dst_folder_path = '/media/neo/92cd53f3-fc9f-4db8-b68e-faf606561e34/home/neo/PycharmProjects/hed_origin/data/new_dataset1/big/augmentation/'
    # edge_image_dst_folder_path = '../../data/new_dataset1/e_edge/augmentation/'
    edge_image_dst_folder_path = '/media/neo/92cd53f3-fc9f-4db8-b68e-faf606561e34/home/neo/PycharmProjects/hed_origin/data/new_dataset1/e_edge/augmentation/'

    orig_image_paths_list = get_image_paths(orig_images_path)
    edge_image_paths_list = get_image_paths(edge_images_path)

    i = 0

    for orig_image_path, edge_image_path in zip(orig_image_paths_list, edge_image_paths_list):
        if i > n:
            break

        orig_image_filename = get_filename_from_path(orig_image_path) + '.jpg'
        edge_image_filename = get_filename_from_path(edge_image_path) + '.jpg'

        orig_image_dst_path = os.path.join(orig_image_dst_folder_path, orig_image_filename)
        edge_image_dst_path = os.path.join(edge_image_dst_folder_path, edge_image_filename)

        copyfile(orig_image_path, orig_image_dst_path)
        copyfile(edge_image_path, edge_image_dst_path)

        i += 1


def show_size_of_images(images_path):
    print('*********show size***********')
    image_paths_list = get_image_paths(images_path)
    for image_path in image_paths_list:
        image = cv2.imread(image_path)
        (h, w) = image.shape[0:2]
        print('*****processing image: %s, orig size: %d, %d' % (image_path, h, w))


def gradient_resize(images_path):
    resized_image_save_dir = os.path.join(images_path, 'resized')
    if not os.path.exists(resized_image_save_dir):
        os.makedirs(resized_image_save_dir)

    image_paths_list = get_image_paths(images_path)
    for image_path in image_paths_list:
        filename = get_filename_from_path(image_path) + '.jpg'
        new_save_path = os.path.join(resized_image_save_dir, filename)
        image = cv2.imread(image_path)
        (h, w) = image.shape[0:2]
        max_l = w if w > h else h

        print('*****processing image: %s, orig size: %d, %d' % (image_path, h, w))

        # if 3563 <= max_l <= 5344:
        #     resized_image = resize_image_by_the_longest_side(image, 1000)
        # elif 1782 <= max_l <= 3562:
        #     resized_image = resize_image_by_the_longest_side(image, 667)
        # elif 891 <= max_l <= 1781:
        #     resized_image = resize_image_by_the_longest_side(image, 444)
        # else:
        #     resized_image = image

        resized_image = resize_image_by_the_longest_side(image, 500)

        cv2.imwrite(new_save_path, resized_image)


def resize_image_by_the_longest_side(image, expected_length_of_the_longest_side):
    (h, w) = image.shape[0:2]
    if w > h:
        new_w = expected_length_of_the_longest_side
        ratio = float(new_w) / float(w)
        # print('ratio:', ratio)
        new_h = int(math.ceil(ratio * h))
        return cv2.resize(image, (new_w, new_h))
    elif h > w:
        new_h = expected_length_of_the_longest_side
        ratio = float(new_h) / float(h)
        # print('ratio:', ratio)
        new_w = int(math.ceil(ratio * w))
        return cv2.resize(image, (new_w, new_h))
    else:
        new_w = expected_length_of_the_longest_side
        new_h = new_w
        return cv2.resize(image, (new_w, new_h))


def get_rectangle_test_set():
    image_path = '../../data/test1/0161_HG_120118_073_E1_N11_rotated20_resized1_5.png'
    path_to_saving_directory = '../../data/test1/e_edge/'
    filename = get_filename_from_path(image_path)
    image = cv2.imread(image_path)
    (h, w) = image.shape[0:2]
    # print(h, w)
    for new_w in range(500, 2000, 100):
        ratio = float(new_w) / float(w)
        new_h = int(ratio * h)
        resized_image = cv2.resize(image, (new_w, new_h))
        save_filename = filename + '_w' + `new_w` + '.jpg'
        cv2.imwrite(os.path.join(path_to_saving_directory, save_filename), resized_image)


def get_square_test_set():
    image_path = '../../data/test1/square/0361_HG_130208_1418_E3_KR_AII_rotated-30_resized1_5.png'
    path_to_saving_directory = '../../data/test1/square/big/'
    filename = get_filename_from_path(image_path)
    image = cv2.imread(image_path)
    (h, w) = image.shape[0:2]


    for i in range(50):
        resized_image = cv2.resize(image, (500, 500))
        save_filename = filename + '_500' + `i` + '.jpg'
        cv2.imwrite(os.path.join(path_to_saving_directory, save_filename), resized_image)

    # for l in range(600, 5300, 200):
    #     resized_image = cv2.resize(image, (l, l))
    #     save_filename = filename + '_' + `l` + '.jpg'
    #     cv2.imwrite(os.path.join(path_to_saving_directory, save_filename), resized_image)


def find_the_biggest_width_from_image_folder(path_to_image_folder):
    image_paths_list = get_image_paths(path_to_image_folder)
    max_l = 0
    for image_path in image_paths_list:
        image = cv2.imread(image_path)
        (h, w) = image.shape[:2]
        if w > h and w > max_l:
            max_l = w
        elif h > w and h > max_l:
            max_l = h
    print('max_w: ', max_l)


def resize_width(images_path):
    resized_image_save_dir = os.path.join(images_path, 'resized')
    if not os.path.exists(resized_image_save_dir):
        os.makedirs(resized_image_save_dir)
    image_paths_list = get_image_paths(images_path)

    for image_path in image_paths_list:
        image = cv2.imread(image_path)
        (h, w) =image.shape[:2]

        print('*****processing image: %s, orig size: %d, %d' % (image_path, h, w))

        filename = get_filename_from_path(image_path) + '.jpg'
        new_save_path = os.path.join(resized_image_save_dir, filename)

        if w > 600:
            # new_w = new_width
            # ratio = float(new_w)/float(w)
            # new_h = int(ratio * h)
            # print('****new width and height: %d, %d ' % (new_w, new_h))
            # resized_image = cv2.resize(image, (new_w, new_h))
            # cv2.imwrite(new_save_path, resized_image)
            new_w = int(float(w) / float(6))
            new_h = int(float(h) / float(6))
            resized_image = cv2.resize(image, (new_w, new_h))
            cv2.imwrite(new_save_path, resized_image)
        else:
            cv2.imwrite(new_save_path, image)


def get_train_pair_list(path_to_original_images, path_to_edge_images, path_to_save_lst_file):
    lst_file = open(path_to_save_lst_file, 'w')

    original_image_paths_list = get_image_paths(path_to_original_images)
    edge_image_paths_list = get_image_paths(path_to_edge_images)

    is_all_right = True

    for path_to_original_image, path_to_edge_image in zip(original_image_paths_list, edge_image_paths_list):
        right_path_to_original_image = path_to_original_image[path_to_original_image.find('big'):]
        right_path_to_edge_image = path_to_edge_image[path_to_edge_image.find('e_edge'):]
        lst_file.write("%s %s\n" % (right_path_to_original_image, right_path_to_edge_image))

    lst_file.close()


def convert_background_to_transparency(images_path):
    image_paths_list = get_image_paths(images_path)
    for image_path in image_paths_list:
        print('******Processing Image: ', image_path)
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image[i][j][0] == 255 and image[i][j][1] == 255 and image[i][j][2] == 255 and image[i][j][3] == 255:
                    image[i][j][3] = 0
        cv2.imwrite(image_path, image)


def rotation_coords(x, y, rotation_angle):
    rotation = (math.pi / 180) * rotation_angle
    u = x * math.cos(rotation) - y * math.sin(rotation)
    v = x * math.sin(rotation) + y * math.cos(rotation)
    return u, v


def translation_matrix(x, y):
    return np.array([[1, 0, x],
                     [0, 1, y],
                     [0, 0, 1]])


def affine_transform_matrix(rotation_angle, shear, sx, sy):
    rotation = (math.pi / 180) * rotation_angle

    return np.array([
        [sx * math.cos(rotation), -sy * math.sin(rotation + shear), 0],
        [sx * math.sin(rotation), sy * math.cos(rotation + shear), 0],
        [0, 0, 1]
    ])


def find_image_in_folder(image_name, path_to_image_folder):
    is_found = False

    image_paths_list = get_image_paths(path_to_image_folder)
    for image_path in image_paths_list:
        if get_filename_from_path(image_path) + '.png' == image_name:
            is_found = True

    return is_found


def data_augmentation(db_path, orig_img_path, edge_img_path, right_image_path):
    '''
    Augment both orignal images and edge images
    :param db_path:
    :param orig_img_path:
    :param edge_img_path:
    :param right_image_path:
    :param transformation_matrix:
    :return:
    '''

    MAX_DIM = 700
    UP_FACTOR = 1.5
    new_max_dim = int(math.floor(MAX_DIM / UP_FACTOR))

    #### Create the directories of augmented images ####
    augmented_edge_image_save_dir = os.path.join(edge_img_path, 'augmentation')
    if not os.path.exists(augmented_edge_image_save_dir):
        os.makedirs(augmented_edge_image_save_dir)
    augmented_orig_image_save_dir = os.path.join(orig_img_path, 'augmentation')
    if not os.path.exists(augmented_orig_image_save_dir):
        os.makedirs(augmented_orig_image_save_dir)

    ##### Get the db file which includes the image information including origName, segmName, and the transformation ####
    ##### function used for matching the original image and segmented image ####
    db = scipy.io.loadmat(db_path, squeeze_me=True)
    db = db['out']

    ##### Traverse the all image from db and augment all the eligible image ####
    for item in db:
        # item -> dict(origName:0002_HG_100429_003_SD.JPG, segmName:0002_HG_100429_003_SD.png, T: 3x3 array)
        original_image_name = item['origName']
        edge_image_name = item['segmName']
        original_image_filename = get_filename_from_path(original_image_name)
        edge_image_filename = get_filename_from_path(edge_image_name)
        # check if the image from db_database is a correct image
        if find_image_in_folder(edge_image_name, right_image_path):
            print('***********Processing Image: ', edge_image_name)
            ##### Read the image using skimage ####
            origI = skimage.io.imread(os.path.join(orig_img_path, original_image_name))
            edgeI = skimage.io.imread(os.path.join(edge_img_path, edge_image_name))

            #### Augment original and edge image ####
            for rotation_angle in range(-60, 80, 20):
                R = affine_transform_matrix(rotation_angle, 0, 1, 1)
                T = item['T']
                T = T.T  # The matrix orientation of python and matlab are different
                sz = edgeI.shape[0:2]  # (x,y) size of the warped image to match size of segmented image
                h, w = sz
                Tr = translation_matrix(-sz[1] / 2, -sz[0] / 2)
                R_center = np.matmul(inv(Tr), np.matmul(R, Tr))
                RT = np.matmul(R_center, T)
                pts_in = [[0, 0],
                          [w - 1, 0],
                          [0, h - 1],
                          [w - 1, h - 1]]
                pts_out = skimage.transform.matrix_transform(pts_in, R_center)
                max_x = max(pts_out[:, 0])
                max_y = max(pts_out[:, 1])
                min_x = min(pts_out[:, 0])
                min_y = min(pts_out[:, 1])
                new_w = math.ceil(max_x - min_x)
                new_h = math.ceil(max_y - min_y)
                sz = (new_h, new_w)
                # print(new_h, new_w)
                RT = np.matmul(translation_matrix(-min_x, -min_y), RT)
                R_center = np.matmul(translation_matrix(-min_x, -min_y), R_center)
                # Generate the rotated image in skimage float format
                skimage_orig_rotated_image = skimage.transform.warp(origI, inv(RT), output_shape=sz)
                skimage_edge_rotated_image = skimage.transform.warp(edgeI, inv(R_center), output_shape=sz)

                #### Generate the augmented original image ####
                # Generate the rotated original image
                skimage_orig_rotated_image = skimage_orig_rotated_image[:, :, ::-1]
                orig_rotated_image = skimage.img_as_ubyte(skimage_orig_rotated_image)
                # Resizes the rotated original images to make sure the length of its longest side is less than MAX_DIM
                (h ,w) = orig_rotated_image.shape[0:2]
                image_max_dim = w if w >= h else h
                if image_max_dim * UP_FACTOR > MAX_DIM:
                    orig_rotated_image = resize_image_by_the_longest_side(orig_rotated_image, new_max_dim)
                # Resizes the rotated original images (scale 0.5 and 1.5)
                dim_0_5 = (int(0.5 * orig_rotated_image.shape[1]), int(0.5 * orig_rotated_image.shape[0]))
                orig_rotated_resized0_5_image = cv2.resize(orig_rotated_image, dim_0_5, interpolation=cv2.INTER_AREA)
                dim_1_5 = (int(1.5 * orig_rotated_image.shape[1]), int(orig_rotated_image.shape[0] * 1.5))
                orig_rotated_resized1_5_image = cv2.resize(orig_rotated_image, dim_1_5, interpolation=cv2.INTER_AREA)
                # Flips rotated and resized original images
                orig_rotated_horizontally_flipped_image = cv2.flip(orig_rotated_image, 0)
                orig_rotated_vertically_flipped_image = cv2.flip(orig_rotated_image, 1)
                orig_rotated_resized0_5_horizontally_flipped_image = cv2.flip(orig_rotated_resized0_5_image, 0)
                orig_rotated_resized0_5_vertically_flipped_image = cv2.flip(orig_rotated_resized0_5_image, 1)
                orig_rotated_resized1_5_horizontally_flipped_image = cv2.flip(orig_rotated_resized1_5_image, 0)
                orig_rotated_resized1_5_vertically_flipped_image = cv2.flip(orig_rotated_resized1_5_image, 1)
                # Generate the filename of the augmented orig images
                orig_rotated_image_name = original_image_filename + '_rotated' + `rotation_angle` + '.png'
                orig_rotated_resized0_5_image_name = original_image_filename + '_rotated' + `rotation_angle` + '_resized0_5' + '.png'
                orig_rotated_resized1_5_image_name = original_image_filename + '_rotated' + `rotation_angle` + '_resized1_5' + '.png'
                orig_rotated_horizontally_flipped_image_name = original_image_filename + '_rotated' + `rotation_angle` + 'horizontally_flipped' + '.png'
                orig_rotated_vertically_flipped_image_name = original_image_filename + '_rotated' + `rotation_angle` + 'vertically_flipped' + '.png'
                orig_rotated_resized0_5_horizontally_flipped_image_name = original_image_filename + '_rotated' + `rotation_angle` + '_resized0_5' + '_horizontally_flipped' + '.png'
                orig_rotated_resized0_5_vertically_flipped_image_name = original_image_filename + '_rotated' + `rotation_angle` + '_resized0_5' + '_horizontally_flipped' + '.png'
                orig_rotated_resized1_5_horizontally_flipped_image_name = original_image_filename + '_rotated' + `rotation_angle` + '_resized1_5' + '_horizontally_flipped' + '.png'
                orig_rotated_resized1_5_vertically_flipped_image_name = original_image_filename + '_rotated' + `rotation_angle` + '_resized1_5' + '_horizontally_flipped' + '.png'
                # Generate the path to save the augmented orig images
                orig_rotated_image_save_path = os.path.join(augmented_orig_image_save_dir, orig_rotated_image_name)
                orig_rotated_resized0_5_image_save_path = os.path.join(augmented_orig_image_save_dir,
                                                                  orig_rotated_resized0_5_image_name)
                orig_rotated_resized1_5_image_save_path = os.path.join(augmented_orig_image_save_dir,
                                                                  orig_rotated_resized1_5_image_name)
                orig_rotated_horizontally_flipped_image_save_path = os.path.join(augmented_orig_image_save_dir,
                                                                            orig_rotated_horizontally_flipped_image_name)
                orig_rotated_vertically_flipped_image_save_path = os.path.join(augmented_orig_image_save_dir,
                                                                          orig_rotated_vertically_flipped_image_name)
                orig_rotated_resized0_5_horizontally_flipped_image_save_path = os.path.join(augmented_orig_image_save_dir,
                                                                                       orig_rotated_resized0_5_horizontally_flipped_image_name)
                orig_rotated_resized0_5_vertically_flipped_image_save_path = os.path.join(augmented_orig_image_save_dir,
                                                                                     orig_rotated_resized0_5_vertically_flipped_image_name)
                orig_rotated_resized1_5_horizontally_flipped_image_save_path = os.path.join(augmented_orig_image_save_dir,
                                                                                       orig_rotated_resized1_5_horizontally_flipped_image_name)
                orig_rotated_resized1_5_vertically_flipped_image_save_path = os.path.join(augmented_orig_image_save_dir,
                                                                                     orig_rotated_resized1_5_vertically_flipped_image_name)
                # Save the augmented orig images
                cv2.imwrite(orig_rotated_image_save_path, orig_rotated_image)
                cv2.imwrite(orig_rotated_resized0_5_image_save_path, orig_rotated_resized0_5_image)
                cv2.imwrite(orig_rotated_resized1_5_image_save_path, orig_rotated_resized1_5_image)
                cv2.imwrite(orig_rotated_horizontally_flipped_image_save_path, orig_rotated_horizontally_flipped_image)
                cv2.imwrite(orig_rotated_vertically_flipped_image_save_path, orig_rotated_vertically_flipped_image)
                cv2.imwrite(orig_rotated_resized0_5_horizontally_flipped_image_save_path,
                            orig_rotated_resized0_5_horizontally_flipped_image)
                cv2.imwrite(orig_rotated_resized0_5_vertically_flipped_image_save_path,
                            orig_rotated_resized0_5_vertically_flipped_image)
                cv2.imwrite(orig_rotated_resized1_5_horizontally_flipped_image_save_path,
                            orig_rotated_resized1_5_horizontally_flipped_image)
                cv2.imwrite(orig_rotated_resized1_5_vertically_flipped_image_save_path,
                            orig_rotated_resized1_5_vertically_flipped_image)

                ##### Generate the augmented edge image ####
                # Generate the rotated edge image
                edge_rotated_image = skimage.img_as_ubyte(skimage_edge_rotated_image)
                # Set the transparent pixel in rotated edge image to white pixel
                for i in range(edge_rotated_image.shape[0]):
                    for j in range(edge_rotated_image.shape[1]):
                        if edge_rotated_image[i][j][3] == 0:
                            edge_rotated_image[i][j][0] = 255
                            edge_rotated_image[i][j][1] = 255
                            edge_rotated_image[i][j][2] = 255
                            edge_rotated_image[i][j][3] = 255
                # Resizes the rotated original images to make sure the length of its longest side is less than MAX_DIM
                (h, w) = edge_rotated_image.shape[0:2]
                image_max_dim = w if w >= h else h
                if image_max_dim * UP_FACTOR > MAX_DIM:
                    edge_rotated_image = resize_image_by_the_longest_side(edge_rotated_image, new_max_dim)
                # Resizes the rotated edge images (scale 0.5 and 1.5)
                dim_0_5 = (int(0.5 * edge_rotated_image.shape[1]), int(0.5 * edge_rotated_image.shape[0]))
                edge_rotated_resized0_5_image = cv2.resize(edge_rotated_image, dim_0_5, interpolation=cv2.INTER_AREA)
                dim_1_5 = (int(1.5 * edge_rotated_image.shape[1]), int(edge_rotated_image.shape[0] * 1.5))
                edge_rotated_resized1_5_image = cv2.resize(edge_rotated_image, dim_1_5, interpolation=cv2.INTER_AREA)
                # Flips rotated and resized edge images
                edge_rotated_horizontally_flipped_image = cv2.flip(edge_rotated_image, 0)
                edge_rotated_vertically_flipped_image = cv2.flip(edge_rotated_image, 1)
                edge_rotated_resized0_5_horizontally_flipped_image = cv2.flip(edge_rotated_resized0_5_image, 0)
                edge_rotated_resized0_5_vertically_flipped_image = cv2.flip(edge_rotated_resized0_5_image, 1)
                edge_rotated_resized1_5_horizontally_flipped_image = cv2.flip(edge_rotated_resized1_5_image, 0)
                edge_rotated_resized1_5_vertically_flipped_image = cv2.flip(edge_rotated_resized1_5_image, 1)
                # Generates the filename of the augmented edge images
                edge_rotated_image_name = edge_image_filename + '_rotated' + `rotation_angle` + '.png'
                edge_rotated_resized0_5_image_name = edge_image_filename + '_rotated' + `rotation_angle` + '_resized0_5' + '.png'
                edge_rotated_resized1_5_image_name = edge_image_filename + '_rotated' + `rotation_angle` + '_resized1_5' + '.png'
                edge_rotated_horizontally_flipped_image_name = edge_image_filename + '_rotated' + `rotation_angle` + 'horizontally_flipped' + '.png'
                edge_rotated_vertically_flipped_image_name = edge_image_filename + '_rotated' + `rotation_angle` + 'vertically_flipped' + '.png'
                edge_rotated_resized0_5_horizontally_flipped_image_name = edge_image_filename + '_rotated' + `rotation_angle` + '_resized0_5' + '_horizontally_flipped' + '.png'
                edge_rotated_resized0_5_vertically_flipped_image_name = edge_image_filename + '_rotated' + `rotation_angle` + '_resized0_5' + '_horizontally_flipped' + '.png'
                edge_rotated_resized1_5_horizontally_flipped_image_name = edge_image_filename + '_rotated' + `rotation_angle` + '_resized1_5' + '_horizontally_flipped' + '.png'
                edge_rotated_resized1_5_vertically_flipped_image_name = edge_image_filename + '_rotated' + `rotation_angle` + '_resized1_5' + '_horizontally_flipped' + '.png'
                # Generate the path to save the augmented edge images
                edge_rotated_image_save_path = os.path.join(augmented_edge_image_save_dir, edge_rotated_image_name)
                edge_rotated_resized0_5_image_save_path = os.path.join(augmented_edge_image_save_dir,
                                                                  edge_rotated_resized0_5_image_name)
                edge_rotated_resized1_5_image_save_path = os.path.join(augmented_edge_image_save_dir,
                                                                  edge_rotated_resized1_5_image_name)
                edge_rotated_horizontally_flipped_image_save_path = os.path.join(augmented_edge_image_save_dir,
                                                                            edge_rotated_horizontally_flipped_image_name)
                edge_rotated_vertically_flipped_image_save_path = os.path.join(augmented_edge_image_save_dir,
                                                                          edge_rotated_vertically_flipped_image_name)
                edge_rotated_resized0_5_horizontally_flipped_image_save_path = os.path.join(augmented_edge_image_save_dir,
                                                                                       edge_rotated_resized0_5_horizontally_flipped_image_name)
                edge_rotated_resized0_5_vertically_flipped_image_save_path = os.path.join(augmented_edge_image_save_dir,
                                                                                     edge_rotated_resized0_5_vertically_flipped_image_name)
                edge_rotated_resized1_5_horizontally_flipped_image_save_path = os.path.join(augmented_edge_image_save_dir,
                                                                                       edge_rotated_resized1_5_horizontally_flipped_image_name)
                edge_rotated_resized1_5_vertically_flipped_image_save_path = os.path.join(augmented_edge_image_save_dir,
                                                                                     edge_rotated_resized1_5_vertically_flipped_image_name)
                # Save the augmented edge images
                cv2.imwrite(edge_rotated_image_save_path, edge_rotated_image)
                cv2.imwrite(edge_rotated_resized0_5_image_save_path, edge_rotated_resized0_5_image)
                cv2.imwrite(edge_rotated_resized1_5_image_save_path, edge_rotated_resized1_5_image)
                cv2.imwrite(edge_rotated_horizontally_flipped_image_save_path, edge_rotated_horizontally_flipped_image)
                cv2.imwrite(edge_rotated_vertically_flipped_image_save_path, edge_rotated_vertically_flipped_image)
                cv2.imwrite(edge_rotated_resized0_5_horizontally_flipped_image_save_path,
                            edge_rotated_resized0_5_horizontally_flipped_image)
                cv2.imwrite(edge_rotated_resized0_5_vertically_flipped_image_save_path,
                            edge_rotated_resized0_5_vertically_flipped_image)
                cv2.imwrite(edge_rotated_resized1_5_horizontally_flipped_image_save_path,
                            edge_rotated_resized1_5_horizontally_flipped_image)
                cv2.imwrite(edge_rotated_resized1_5_vertically_flipped_image_save_path,
                            edge_rotated_resized1_5_vertically_flipped_image)


def rotation_transformation():
    db = scipy.io.loadmat('../../data/test/out.mat', squeeze_me=True)
    db = db['out']
    # paths to original images and to segmented images
    orig_img_path = '../../data/new_dataset/big/'
    segm_img_path = '../../data/new_dataset/crop/'
    edge_img_path = '../../data/new_dataset/crop/gt_boundary/'
    f = db[2000]

    # f -> dict(origName:0002_HG_100429_003_SD.JPG, segmName:0002_HG_100429_003_SD.png, T: 3x3 array)
    origI = skimage.io.imread(os.path.join(orig_img_path, f['origName']))
    # segmI = skimage.io.imread(os.path.join(segm_img_path, f['segmName']))
    edgeI = skimage.io.imread(os.path.join(edge_img_path, f['segmName']))

    R = affine_transform_matrix(10, 0, 1, 1)

    T = f['T']
    T = T.T  # The matrix orientation of python and matlab are different


    # RTinv = inv(RT)  # warp function requires inverse transfrom

    sz = edgeI.shape[0:2]  # (x,y) size of the warped image to match size of segmented image
    h,w = sz

    Tr = translation_matrix(-sz[1]/2,-sz[0]/2)
    Trback = inv(Tr)

    # R_center = np.matmul(inv(Tr), Tr)
    R_center = np.matmul(inv(Tr), np.matmul(R, Tr))

    RT = np.matmul(R_center, T)

    pts_in = [[0  , 0],
              [w-1, 0],
              [0  , h-1],
              [w-1, h-1]]

    pts_out= skimage.transform.matrix_transform(pts_in,R_center)

    max_x = max(pts_out[:,0])
    max_y = max(pts_out[:,1])
    min_x = min(pts_out[:,0])
    min_y = min(pts_out[:,1])
    new_w = math.ceil(max_x - min_x)
    new_h = math.ceil(max_y - min_y)
    sz = (new_h, new_w)

    print(new_h,new_w)

    RT = np.matmul(translation_matrix(-min_x,-min_y),RT)
    R_center = np.matmul(translation_matrix(-min_x,-min_y),R_center)
    # sz = (785, 1040)

    segmI1 = skimage.transform.warp(origI, inv(T), output_shape=sz)
    segmI2 = skimage.transform.warp(origI, inv(RT), output_shape=sz)
    edgeI2 = skimage.transform.warp(edgeI, inv(R_center), output_shape=sz)
    skimage.io.imsave('../../data/test/result1.png', segmI1)
    skimage.io.imsave('../../data/test/result2.png', segmI2)
    skimage.io.imsave('../../data/test/rotated_edge.png', edgeI2)

    return segmI2, edgeI2


def read_images_list_from_file(lst_file_path):
    f = open(lst_file_path, 'r')
    images_list = []
    images_list_content = f.readlines()
    for image_name in images_list_content:
        print(image_name.rstrip())
        images_list.append(image_name.rstrip())
    f.close()

    print('****Images: ', images_list)
    print('****length: ', len(images_list))
    return images_list


def get_filename_from_path(filepath):
    path, filename = os.path.split(filepath)
    filename = os.path.splitext(filename)[0]
    return filename


def begin_with_dot(path):
    split_path = path.split('/')
    if split_path[-1].startswith('._'):
        # print(split_path[-1])
        return True
    return False


def rotate_image(image, angle):
    """
    Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
    (in degrees). The returned image will be large enough to hold the entire
    new image, with a black background
    """

    # Get the image size
    # No that's not an error - NumPy stores image matricies backwards
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack(
        [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
    )

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]
    ])

    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

    # Apply the transform
    result = cv2.warpAffine(
        image,
        affine_mat,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR
    )

    return result


def largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell
    """

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )


def crop_around_center(image, width, height):
    """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point
    """

    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if(width > image_size[0]):
        width = image_size[0]

    if(height > image_size[1]):
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]


def get_rotation_image(img_path, rotation_angle):
    image = cv2.imread(img_path)
    image_height, image_width = image.shape[0:2]

    image_rotated = rotate_image(image, rotation_angle)
    image_rotated_cropped = crop_around_center(
        image_rotated,
        *largest_rotated_rect(
            image_width,
            image_height,
            math.radians(rotation_angle)
        )
    )
    # Returns the rotated and cropped image
    return image_rotated


def get_fin_boundary_image(img_path):
    problem = False

    # Read the image
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    # Using Otsu's thresholding to get the threshold which can be used for the first removal of noise
    gray_orig_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    threshold_value, thresholded_img = cv2.threshold(gray_orig_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print('Threshold Value: ', threshold_value)

    # The first removal of noise using threshold value
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j][3] > threshold_value:
                img[i][j][3] = 255
                img[i][j][0] = 255
                img[i][j][1] = 255
                img[i][j][2] = 255
            else:
                img[i][j][3] = 255
                img[i][j][0] = 0
                img[i][j][1] = 0
                img[i][j][2] = 0

    # The second removal of noise (keep the biggest contour of the image)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert the processed image to gray scale image

    try:
        (height, width) = gray_img.shape[:2]
        fin_img = np.zeros((height, width, 4), np.uint8) # initialise the fin_img as a transparent image
        im2, contours, hierarchy = cv2.findContours(gray_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        largest_area = 0
        largest_area_index = 0
        for i in range(len(contours)): # find the largest contour in the image
            contour_area = cv2.contourArea(contours[i])
            if contour_area > largest_area:
                largest_area = contour_area
                largest_area_index = i
        cnt = contours[largest_area_index]
        cv2.drawContours(fin_img, [cnt], -1, (255, 255, 255), -1) # draw the extracted fin on the transparent image
        for i in range(fin_img.shape[0]):
            for j in range(fin_img.shape[1]):
                # if the pixel is white (fin is coloured by white), set the alpha channel of this pixel to 255
                # Also, set the color of the fin area to red (BGR)
                if fin_img[i][j][0] == 255 and fin_img[i][j][1] == 255 and fin_img[i][j][2] == 255:
                    fin_img[i][j][3] = 255
                    fin_img[i][j][0] = 0
                    fin_img[i][j][1] = 0
                    fin_img[i][j][2] = 255

        # Contour extraction using morphological operation
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)) # constructs a 3 x 3 element
        dilate = cv2.dilate(fin_img, element, iterations=1)
        erode = cv2.erode(fin_img, element)
        result = cv2.absdiff(dilate, erode) # subtracts eroded image from dilated image to get the boundary

        # Covert the transparent pixel to white pixel
        # for i in range(result.shape[0]):
        #     for j in range(result.shape[1]):
        #         if result[i][j][3] == 0:
        #             result[i][j][3] = 255
        #             result[i][j][0] = 255
        #             result[i][j][1] = 255
        #             result[i][j][2] = 255
    except BaseException:
        # Catch error of cv2.findContours(), return False
        print('****cv2.findContours() Error!****')
        problem = True
        result = -1
        return problem, result
    else:
        # Return the result (fin-contour image)
        return problem, result


def get_image_names(dir_path):
    image_paths = get_image_paths(dir_path)
    image_names = []

    for image_path in image_paths:
        path, filename = os.path.split(image_path)
        image_name = os.path.splitext(filename)[0]
        image_names.append(image_name)

    return image_names


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


def get_fin_boundary_from_dir(input_dir_path, boundary_save_path):
    '''
    Saves the images of extracted boundary to the target path. The overlay image is the overlay the
    extracted boundary on the original input image.
    :param input_dir_path: The path to input images
    :param boundary_save_path: The path to save the images of extracted boundary
    :return: None
    '''
    input_image_paths = get_image_paths(input_dir_path)
    problematic_image_paths = []
    for input_path in input_image_paths:
        print('********Processing boundary: ', input_path)
        problem, result = get_fin_boundary_image(input_path)
        if not problem:
            path, filename = os.path.split(input_path)
            output_path = os.path.join(boundary_save_path, filename)
            cv2.imwrite(output_path, result)
        else:
            print('****Problematic Image: ', input_path)
            # Adds the problematic image to array problematic_image_paths
            problematic_image_paths.append(input_path)
            continue

    print('Problematic Images: ', problematic_image_paths)
    output_problematic_image_lst_file_from_list(problematic_image_paths, os.path.join(boundary_save_path, 'problematic_images.lst'))
    return problematic_image_paths


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
    # image2 = cv2.bitwise_not(image2)
    overlay = image1
    output = image2
    cv2.addWeighted(overlay, alpha, output, 1, 0, output)
    output_path = os.path.join(overlay_save_path, filename)
    print('********Processing overlay: ', output_path)
    cv2.imwrite(output_path, output)


def overlay_edge_images_on_orignal_images(original_images_path, edge_images_path):
    overlay_images_save_path = os.path.join(original_images_path, 'overlay')
    if not os.path.exists(overlay_images_save_path):
        os.makedirs(overlay_images_save_path)

    edge_image_path_list = get_image_paths(edge_images_path)
    original_image_path_list = get_image_paths(original_images_path)

    for edge_image_path in edge_image_path_list:
        image_name = get_filename_from_path(edge_image_path)
        corresponding_original_image_path = ''

        for original_image_path in original_image_path_list:
            original_image_name = get_filename_from_path(original_image_path)
            if original_image_name == image_name:
                corresponding_original_image_path = original_image_path

        edge_image = cv2.imread(edge_image_path)
        original_image = cv2.imread(corresponding_original_image_path)

        overlay_image_filename_with_ext = image_name + '.png'

        get_overlay_image(edge_image, original_image, overlay_images_save_path, overlay_image_filename_with_ext)
        # get_overlay_image(original_image, edge_image, overlay_images_save_path, overlay_image_filename_with_ext)


def get_ground_truth(root_path):
    boundary_save_path = os.path.join(root_path, 'gt_boundary')
    if not os.path.exists(boundary_save_path):
        os.makedirs(boundary_save_path)

    get_fin_boundary_from_dir(root_path, boundary_save_path)


def get_image_paths(dir_path):
    """ Returns a list of paths to JPG and PNG files in dir_path. """
    regex = re.compile(r'.*\.(jpe?g|png)$', re.IGNORECASE)
    paths = [os.path.join(dir_path, f) for f in os.listdir(dir_path)
             if os.path.isfile(os.path.join(dir_path, f)) and regex.match(f)]

    new_paths = [x for x in paths if not begin_with_dot(x)]

    return new_paths


def output_problematic_image_lst_file_from_list(image_paths, output_image_lst_file_path):
    # list_path = '../../data/segmentation_fin.lst'
    list_path = output_image_lst_file_path
    list_file = open(list_path, 'w')
    for path in image_paths:
        npath = path[path.find('crop')+5:]
        list_file.write("%s\n" % npath)


def parse_args(argv):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('cmd', choices=['get_lst_file_from_dir', 'get_trainpair_lst_file', 'data_augmentation', 'get_ground_truth', 'overlay', 'test'],
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
    output_image_lst_file_path = args.output_image_lst_file_path
    output_trainpair_lst_file_path = args.output_trainpair_lst_file_path
    output_augmented_images_path = args.output_augmented_images_path

    if cmd == 'get_lst_file_from_dir':
        print('1')
        # Outputs a lst file which contains the paths of every images in the directory input_image_path
        # to output_image_lst_file_path
        # output_lst_file_from_list(image_paths, output_image_lst_file_path)
    elif cmd == 'get_trainpair_lst_file':
        get_train_pair_list('../../data/new_dataset/big/augmentation/', '../../data/new_dataset/e_edge/augmentation/',
                            '../../data/new_dataset/train_pair.lst')
    elif cmd == 'data_augmentation':
        print('data_augmentation')
        db_path = '../../data/new_dataset/out.mat'
        orig_img_path = '../../data/new_dataset/big/'
        edge_img_path = '../../data/new_dataset/e_edge/'
        right_image_path = '../../data/new_dataset/e_edge/'
        data_augmentation(db_path, orig_img_path, edge_img_path, right_image_path)
    elif cmd == 'get_ground_truth':
        get_ground_truth(input_image_path)
    elif cmd == 'overlay':
        overlay_edge_images_on_orignal_images('../../data/new_dataset/orig/',
                                              '../../data/new_dataset/crop/gt_boundary/')
    elif cmd == 'test':
        print('test')
        # get_rotation_image('../../data/test_rotation/0006_HG_120601_215_E3_LH_rotation0.png', -10)
        # overlay_edge_images_on_orignal_images('../../data/new_dataset/big/augmentation', '../../data/new_dataset/e_edge/augmentation')
        # test(input_image_path)
        # read_images_list_from_file('../../data/new_dataset/crop/gt_boundary/problematic_images.txt')
        # test('../../data/new_dataset/data_cleaning/', '../../data/new_dataset/orig/overlay')
        # transformation()
        # test()
        # rotation_matrix_3(10)
        # translation((0,0), (0,1), (1,0), (1,1))
        # data_augmentation_test()


if __name__ == '__main__':
    main(sys.argv[1:])