from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from PIL import Image
import Augmentor
import imutils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def copy_image(path_to_problematic_images_dir, path_to_images_dir):
    problematic_image_names_list = get_image_names(path_to_problematic_images_dir)

    image_paths_list = get_image_paths(path_to_images_dir)

    problematic_orig_image_paths_list = []
    problematic_crop_image_paths_list = []
    problematic_edge_image_paths_list = []
    problematic_overlay_image_paths_list = []

    for problematic_image_name in problematic_image_names_list:
        path_to_problematic_orig_image = '../../data/new_dataset/orig/' + problematic_image_name + '.JPG'
        path_to_problematic_crop_image = '../../data/new_dataset/crop/' + problematic_image_name + '.png'
        path_to_problematic_edge_image = '../../data/new_dataset/crop/gt_boundary/' + problematic_image_name + '.png'
        path_to_problematic_overlay_image = '../../data/new_dataset/orig/overlay/' + problematic_image_name + '.png'

        problematic_orig_image_paths_list.append(path_to_problematic_orig_image)
        problematic_crop_image_paths_list.append(path_to_problematic_crop_image)
        problematic_edge_image_paths_list.append(path_to_problematic_edge_image)
        problematic_overlay_image_paths_list.append(path_to_problematic_overlay_image)

    right_image_paths_list = [item for item in image_paths_list if item not in problematic_overlay_image_paths_list]

    for right_image_path in right_image_paths_list:
        right_image_name = get_filename_from_path(right_image_path)
        right_image_name += '.png'
        right_image_dst_path = os.path.join('../../data/new_dataset/e_overlay', right_image_name)
        copyfile(right_image_path, right_image_dst_path)


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
            # Produces the augmented images
            rotated_image = get_rotation_image(image, rotation_angle)
            # rotated_image, rotated_edge_image = rotation_transformation()
            # Resizes the rotated images (scale 0.5 and 1.5)
            dim_0_5 = (int(0.5 * rotated_image.shape[1]), int(0.5 * rotated_image.shape[0]))
            rotated_resized0_5_image = cv2.resize(rotated_image, dim_0_5, interpolation=cv2.INTER_AREA)
            dim_1_5 = (int(1.5 * rotated_image.shape[1]), int(rotated_image.shape[0] * 1.5))
            rotated_resized1_5_image = cv2.resize(rotated_image, dim_1_5, interpolation=cv2.INTER_AREA)
            # Flips rotated and resized images
            rotated_horizontally_flipped_image = cv2.flip(rotated_image, 0)
            rotated_vertically_flipped_image = cv2.flip(rotated_image, 1)
            rotated_resized0_5_horizontally_flipped_image = cv2.flip(rotated_resized0_5_image, 0)
            rotated_resized0_5_vertically_flipped_image= cv2.flip(rotated_resized0_5_image, 1)
            rotated_resized1_5_horizontally_flipped_image = cv2.flip(rotated_resized1_5_image, 0)
            rotated_resized1_5_vertically_flipped_image = cv2.flip(rotated_resized1_5_image, 1)

            # Saves augmented images
            path, filename = os.path.split(image)
            filename = os.path.splitext(filename)[0]
            # Generates the filename of the augmented images
            rotated_image_name = filename + '_rotated' + `rotation_angle` + '.png'
            rotated_resized0_5_image_name = filename + '_rotated' + `rotation_angle` + '_resized0_5' + '.png'
            rotated_resized1_5_image_name = filename + '_rotated' + `rotation_angle` + '_resized1_5' + '.png'
            rotated_horizontally_flipped_image_name = filename + '_rotated' + `rotation_angle` + 'horizontally_flipped' + '.png'
            rotated_vertically_flipped_image_name = filename + '_rotated' + `rotation_angle` + 'vertically_flipped' + '.png'
            rotated_resized0_5_horizontally_flipped_image_name = filename + '_rotated' + `rotation_angle` + '_resized0_5' + '_horizontally_flipped' + '.png'
            rotated_resized0_5_vertically_flipped_image_name = filename + '_rotated' + `rotation_angle` + '_resized0_5' + '_horizontally_flipped' + '.png'
            rotated_resized1_5_horizontally_flipped_image_name = filename + '_rotated' + `rotation_angle` + '_resized1_5' + '_horizontally_flipped' + '.png'
            rotated_resized1_5_vertically_flipped_image_name = filename + '_rotated' + `rotation_angle` + '_resized1_5' + '_horizontally_flipped' + '.png'
            # Generates the path of the augmented images
            rotated_image_save_path = os.path.join(augmented_image_save_dir, rotated_image_name)
            rotated_resized0_5_image_save_path = os.path.join(augmented_image_save_dir, rotated_resized0_5_image_name)
            rotated_resized1_5_image_save_path = os.path.join(augmented_image_save_dir, rotated_resized1_5_image_name)
            rotated_horizontally_flipped_image_save_path = os.path.join(augmented_image_save_dir, rotated_horizontally_flipped_image_name)
            rotated_vertically_flipped_image_save_path = os.path.join(augmented_image_save_dir, rotated_vertically_flipped_image_name)
            rotated_resized0_5_horizontally_flipped_image_save_path = os.path.join(augmented_image_save_dir, rotated_resized0_5_horizontally_flipped_image_name)
            rotated_resized0_5_vertically_flipped_image_save_path = os.path.join(augmented_image_save_dir, rotated_resized0_5_vertically_flipped_image_name)
            rotated_resized1_5_horizontally_flipped_image_save_path = os.path.join(augmented_image_save_dir, rotated_resized1_5_horizontally_flipped_image_name)
            rotated_resized1_5_vertically_flipped_image_save_path = os.path.join(augmented_image_save_dir, rotated_resized1_5_vertically_flipped_image_name)

            cv2.imwrite(rotated_image_save_path, rotated_image)
            cv2.imwrite(rotated_resized0_5_image_save_path, rotated_resized0_5_image)
            cv2.imwrite(rotated_resized1_5_image_save_path, rotated_resized1_5_image)
            cv2.imwrite(rotated_horizontally_flipped_image_save_path, rotated_horizontally_flipped_image)
            cv2.imwrite(rotated_vertically_flipped_image_save_path, rotated_vertically_flipped_image)
            cv2.imwrite(rotated_resized0_5_horizontally_flipped_image_save_path, rotated_resized0_5_horizontally_flipped_image)
            cv2.imwrite(rotated_resized0_5_vertically_flipped_image_save_path, rotated_resized0_5_vertically_flipped_image)
            cv2.imwrite(rotated_resized1_5_horizontally_flipped_image_save_path, rotated_resized1_5_horizontally_flipped_image)
            cv2.imwrite(rotated_resized1_5_vertically_flipped_image_save_path, rotated_resized1_5_vertically_flipped_image)


def check_orig_crop_resolution():
    crop_image_paths = get_image_paths('../../data/new_dataset/crop')
    orig_image_paths = get_image_paths('../../data/new_dataset/orig')

    problematic_images_paths = []

    for crop_image_path in crop_image_paths:
        path, filename = os.path.split(crop_image_path)
        crop_image_name = os.path.splitext(filename)[0]
        orig_image_path = ""

        for image_path in orig_image_paths:
            if image_path.find(crop_image_name) >= 0:
                orig_image_path = image_path

        crop_image = cv2.imread(crop_image_path)
        orig_image = cv2.imread(orig_image_path)

        (h_crop, w_crop) = crop_image.shape[:2]
        (h_orig, w_orig) = orig_image.shape[:2]

        print('crop image: ', crop_image_path)
        print('orig image: ', orig_image_path)
        print('h_crop, w_crop are %d, %d' % (h_crop, w_crop))
        print('h_orig, w_orig are %d, %d' % (h_orig, w_orig))

        if(h_crop != h_orig) or (w_crop != w_orig):
            problematic_images_paths.append(crop_image_path)

    print('The number of problematic images is: ', len(problematic_images_paths))



def get_right_original_images(root_directories, segmentation_directories):
    right_original_image_paths = []
    nonempty_segmented_directories = []

    empty_directories = get_empty_directories(root_directories)
    nonempty_original_directories = [x for x in root_directories if x not in empty_directories]

    for nonempty_original_directory in nonempty_original_directories:
        nonempty_segmented_directory = os.path.join(nonempty_original_directory, 'PNG')
        nonempty_segmented_directories.append(nonempty_segmented_directory)

    # print('*********length of right original directories: ', len(nonempty_original_directories))
    # print('*********length of right segmented directories: ', len(nonempty_segmented_directories))

    for root_directory, segmentation_directory in zip(nonempty_original_directories, nonempty_segmented_directories):
        original_image_names = get_image_names(root_directory)
        segmented_image_names = get_image_names(segmentation_directory)

        print('*********original images: ', original_image_names)
        print('*********segmented images: ', segmented_image_names)

        union_image_names = list(set(original_image_names).intersection(set(segmented_image_names)))
        print('*********union images: ', union_image_names)

        original_image_paths = get_image_paths(root_directory)
        segmented_image_paths = get_image_paths(segmentation_directory)

        for image_name in union_image_names:
            original_image_path = ''
            segmented_image_path = ''

            for path in original_image_paths:
                if path.find(image_name) >= 0:
                    original_image_path = path

            for path in segmented_image_paths:
                if path.find(image_name) >= 0:
                    segmented_image_path = path

            print(original_image_path)
            print(segmented_image_path)

            original_image = cv2.imread(original_image_path)
            segmented_image = cv2.imread(segmented_image_path)

            (original_height, original_width) = original_image.shape[:2]
            (segmented_height, segmented_width) = segmented_image.shape[:2]

            print('original resolution: %d, %d' % (original_height, original_width))
            print('segmented resolution: %d, %d' % (segmented_height, segmented_width))

            if (original_height == segmented_height) and (original_width == segmented_width):
                right_original_image_paths.append(original_image_path)
                print('right image: ', original_image_path)

    print('The number of right images is: ', len(right_original_image_paths))



def get_missing_segmentation_images(root_directories, segmentation_directories):
    missing_segmentation_original_images = []

    for root_directory, segmentation_directory in zip(root_directories, segmentation_directories):
        # print('root directory: ', root_directory)
        # print('segmentation directory: ', segmentation_directory)
        original_images = get_image_names(root_directory)
        segmented_images = get_image_names(segmentation_directory)
        # print('Original images: ', original_images)
        # print('Segmented images: ', segmented_images)

        missing_segmentation_original_image_names = list(set(original_images).difference(set(segmented_images)))
        if len(missing_segmentation_original_image_names) > 0:
            # print('The directory that have the missing segmented images is: ', root_directory)
            # print('The original images that missing segmented images are: ', missing_segmentation_original_image_names)
            for image_name in missing_segmentation_original_image_names:
                image_name += '.JPG'
                image_path = os.path.join(root_directory, image_name)
                missing_segmentation_original_images.append(image_path)

    return missing_segmentation_original_images


def get_empty_directories(directories):
    empty_directories = []

    for directory in directories:
        if len(get_image_paths(directory)) == 0:
            empty_directories.append(directory)
        # print('Processing directory: ', directory)
        # print('It has %d images' % len(get_image_paths(directory)))

    # print('The number of empty directories is: ', len(empty_directories))
    # print('They are: ', empty_directories)

    return empty_directories


def get_problematic_directories_images(root_directories, segmentation_directories):
    empty_directories = get_empty_directories(root_directories)
    missing_segmentation_original_images = get_missing_segmentation_images(root_directories, segmentation_directories)

    print('***********empty_directories: ', empty_directories)
    print('***********missing_segmentation_original_images: ', missing_segmentation_original_images)

def test_boundary():
    # image = cv2.imread('../../data/test_fin_boundary/test1.png', cv2.IMREAD_UNCHANGED)
    image = np.array(Image.open('../../data/test_fin_boundary/test1.png'))

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j][3] > 225:
                # image[i][j][3] = 255
                image[i][j][0] = 0
                image[i][j][1] = 0
                image[i][j][2] = 0
            # else:
            #     image[i][j][3] = 0

    # constructs a 3 x 3 element
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilate = cv2.dilate(image, element, iterations=2)
    erode = cv2.erode(image, element)

    # subtracts eroded image from dilated image to get the boundary
    result = cv2.absdiff(dilate, erode)

    # binarise grayscale image
    retval, result = cv2.threshold(result, 40, 255, cv2.THRESH_BINARY)

    result_black = result.copy()

    for i in range(result_black.shape[0]):
        for j in range(result_black.shape[1]):
            if result_black[i][j][3] == 0:
                result_black[i][j][3] = 255
                result_black[i][j][0] = 0
                result_black[i][j][1] = 0
                result_black[i][j][2] = 0
            elif (result_black[i][j][3] != 0) and (
                    (result_black[i][j][0] != 0) or (result_black[i][j][1] != 0) or (result_black[i][j][2] != 0)):
                result_black[i][j][0] = 255
                result_black[i][j][1] = 255
                result_black[i][j][2] = 255

    cv2.imwrite('../../data/test_fin_boundary/fin_boundary.png', result_black)


def get_list_from_multilevelDir(input_image_path):
    # root_directory = '../../data/Final_Database'
    root_directory = input_image_path

    folder_names = []
    sub_directories = []
    image_paths = []
    first_level_subdirectories = []

    for root, dirs, files in os.walk(root_directory):
        if root == root_directory:
            folder_names = dirs
        # print('****root****: ', root)
        # print('****dirs****: ', dirs)
        # print('****files****: ', files)

    # print(folder_names)
    # print(len(folder_names))
    for folder_name in folder_names:
        path = os.path.join(root_directory, folder_name)
        # path = os.path.join(path, '/PNG')
        first_level_subdirectories.append(path)
        path = path + '/PNG'
        sub_directories.append(path)

    # print(sub_directories)

    for directory in sub_directories:
        image_paths.extend(get_image_paths(directory))

    # print('*********************image paths**********************')
    # print(image_paths)

    return first_level_subdirectories, sub_directories, image_paths




def data_augmentation_test():
    rotation_angle = 30

    img = cv2.imread('../../data/test_rotation/test.JPG')
    (height, width) = img.shape[:2]
    print('width, height:', width, height)
    center = (width / 2, height / 2)
    print('Coordinates of center:', center)
    hr, wr = rotatedRectWithMaxArea(width, height, 30)
    wr = int(wr)
    hr = int(hr)
    print('wr, hr: ', wr, hr)
    print('crop: ', int(width/2-wr/2), int(width/2+wr/2), int(height/2-hr/2), int(height/2+hr/2))

    M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    rotated = cv2.warpAffine(img, M, (width, height), flags=cv2.INTER_LINEAR)

    ma_image = rotated[int(width/2-wr/2):int(width/2+wr/2), int(height/2-hr/2):int(height/2+hr/2)]
    # ma_image = rotated[int(height/2-hr/2):int(height/2+hr/2), int(width/2-wr/2):int(width/2+wr/2)]

    cv2.imwrite('../../data/test_rotation/rotated_30.jpg', rotated)
    # cv2.imwrite('../../data/test_rotation/test_result.jpg', ma_image)


def rotatedRectWithMaxArea(w, h, angle):
  """
  Given a rectangle of size wxh that has been rotated by 'angle' (in
  radians), computes the width and height of the largest possible
  axis-aligned rectangle (maximal area) within the rotated rectangle.
  """
  if w <= 0 or h <= 0:
    return 0,0

  width_is_longer = w >= h
  side_long, side_short = (w,h) if width_is_longer else (h,w)

  # since the solutions for angle, -angle and 180-angle are all the same,
  # if suffices to look at the first quadrant and the absolute values of sin,cos:
  sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
  if side_short <= 2.*sin_a*cos_a*side_long or abs(sin_a-cos_a) < 1e-10:
    # half constrained case: two crop corners touch the longer side,
    #   the other two corners are on the mid-line parallel to the longer line
    x = 0.5*side_short
    wr,hr = (x/sin_a,x/cos_a) if width_is_longer else (x/cos_a,x/sin_a)
  else:
    # fully constrained case: crop touches all 4 sides
    cos_2a = cos_a*cos_a - sin_a*sin_a
    wr,hr = (w*cos_a - h*sin_a)/cos_2a, (h*cos_a - w*sin_a)/cos_2a

  return wr,hr




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



def data_augmentation_edge_images(root_path):
    # for directory in directories:
    images_list = get_image_paths(root_path)

    augmented_image_save_dir = os.path.join(root_path, 'augmentation')
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

            for i in range(rotated.shape[0]):
                for j in range(rotated.shape[1]):
                    if rotated[i][j][3] == 0:
                        rotated[i][j][3] = 255
                        rotated[i][j][0] = 255
                        rotated[i][j][1] = 255
                        rotated[i][j][2] = 255

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

            for i in range(rotated.shape[0]):
                for j in range(rotated.shape[1]):
                    if rotated[i][j][3] == 0:
                        rotated[i][j][3] = 255
                        rotated[i][j][0] = 255
                        rotated[i][j][1] = 255
                        rotated[i][j][2] = 255

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
