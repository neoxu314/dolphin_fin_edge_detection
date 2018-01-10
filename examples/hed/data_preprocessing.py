'''
Data preprocessing for fin images
'''

import cv2
import sys
import os
import re
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def test():
    edge_image = get_fin_boundary_image('../../data/new_dataset/crop/0100_HG_130208_1302_E3_KR_AII.png')
    cv2.imwrite('../../data/new_dataset/crop/0100_HG_130208_1302_E3_KR_AII_edge.png', edge_image)


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

    return image_rotated_cropped


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
                if fin_img[i][j][0] == 255 and fin_img[i][j][1] == 255 and fin_img[i][j][2] == 255:
                    fin_img[i][j][3] = 255
                    fin_img[i][j][0] = 0
                    fin_img[i][j][1] = 0
                    fin_img[i][j][2] = 0

        # Contour extraction using morphological operation
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)) # constructs a 3 x 3 element
        dilate = cv2.dilate(fin_img, element, iterations=1)
        erode = cv2.erode(fin_img, element)
        result = cv2.absdiff(dilate, erode) # subtracts eroded image from dilated image to get the boundary

        # Covert the transparent pixel to white pixel
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                if result[i][j][3] == 0:
                    result[i][j][3] = 255
                    result[i][j][0] = 255
                    result[i][j][1] = 255
                    result[i][j][2] = 255
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
    Saves the overlay images and images of extracted boundary to the target path. The overlay image is the overlay the
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
    alpha = 0.4
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


def data_augmentation(root_path):
    # for directory in directories:
    images_list = get_image_paths(root_path)

    augmented_image_save_dir = os.path.join(root_path, 'augmentation')
    print('save augmented image: ', augmented_image_save_dir)
    if not os.path.exists(augmented_image_save_dir):
        os.makedirs(augmented_image_save_dir)

    for image in images_list:
        print('****************processing augmentation of image: ', image)
        for rotation_angle in range(-10, 20, 10):
            rotated = get_rotation_image(image, rotation_angle)

            # resize
            dim_0_5 = (int(0.5 * rotated.shape[1]), int(0.5 * rotated.shape[0]))
            resized_0_5 = cv2.resize(rotated, dim_0_5, interpolation=cv2.INTER_AREA)
            dim_1_5 = (int(1.5 * rotated.shape[1]), int(rotated.shape[0] * 1.5))
            resized_1_5 = cv2.resize(rotated, dim_1_5, interpolation=cv2.INTER_AREA)

            path, filename = os.path.split(image)
            filename = os.path.splitext(filename)[0]

            rotated_image_name = filename + '_rotation' + `rotation_angle` + '.png'
            rotated_image_name_0_5 = filename + '_rotation' + `rotation_angle` + '_zoom0_5' + '.png'
            rotated_image_name_1_5 = filename + '_rotation' + `rotation_angle` + '_zoom1_5' + '.png'

            rotated_image_save_path = os.path.join(augmented_image_save_dir, rotated_image_name)
            rotated_image_save_path_0_5 = os.path.join(augmented_image_save_dir, rotated_image_name_0_5)
            rotated_image_save_path_1_5 = os.path.join(augmented_image_save_dir, rotated_image_name_1_5)

            cv2.imwrite(rotated_image_save_path, rotated)
            cv2.imwrite(rotated_image_save_path_0_5, resized_0_5)
            cv2.imwrite(rotated_image_save_path_1_5, resized_1_5)




def parse_args(argv):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('cmd', choices=['get_lst_file_from_dir', 'get_trainpair_lst_file', 'data_augmentation', 'get_ground_truth', 'test'],
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
        # Outputs a lst file which contains the paths of every images in the directory input_image_path
        # to output_image_lst_file_path
        output_lst_file_from_list(image_paths, output_image_lst_file_path)
    elif cmd == 'get_trainpair_lst_file':
        print('get_trainpair_lst_file')
    elif cmd == 'data_augmentation':
        print('data_augmentation')
        data_augmentation(input_image_path)
    elif cmd == 'data_augmentation_original_images':
        data_augmentation_orig_images(input_image_path)
    elif cmd == 'get_ground_truth':
        get_ground_truth(input_image_path)
    elif cmd == 'test':
        # get_rotation_image('../../data/test_rotation/0006_HG_120601_215_E3_LH_rotation0.png', -10)
        # overlay_edge_images_on_orignal_images('../../data/new_dataset_test/orig', '../../data/new_dataset_test/crop/gt_boundary')
        test()

if __name__ == '__main__':
    main(sys.argv[1:])