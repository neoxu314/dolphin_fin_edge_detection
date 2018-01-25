import re
import os
from shutil import copyfile


def is_same():
    result = True

    orig_images_path = '/media/neo/92cd53f3-fc9f-4db8-b68e-faf606561e34/home/neo/PycharmProjects/hed_origin/data/new_dataset1/big/augmentation/'
    edge_images_path = '/media/neo/92cd53f3-fc9f-4db8-b68e-faf606561e34/home/neo/PycharmProjects/hed_origin/data/new_dataset1/e_edge/augmentation/'

    orig_image_paths_list = get_image_paths(orig_images_path)
    edge_image_paths_list = get_image_paths(edge_images_path)

    for orig_image_path, edge_image_path in zip(orig_image_paths_list, edge_image_paths_list):
        print(orig_image_path, edge_image_path)
        orig_image_filename = get_filename_from_path(orig_image_path) + '.jpg'
        edge_image_filename = get_filename_from_path(edge_image_path) + '.jpg'

        if orig_image_filename != edge_image_filename:
            result = False

    if result:
        print('Same!')
    else:
        print('Nope!')


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


def get_filename_from_path(filepath):
    path, filename = os.path.split(filepath)
    filename = os.path.splitext(filename)[0]
    return filename


def choose_n_images(n):
    orig_images_path = '../../data/new_dataset/big/augmentation'
    # orig_images_path = '/media/neo/92cd53f3-fc9f-4db8-b68e-faf606561e34/home/neo/PycharmProjects/hed_origin/data/new_dataset/big/augmentation/'
    edge_images_path = '../../data/new_dataset/e_edge/augmentation'
    # edge_images_path = '/media/neo/92cd53f3-fc9f-4db8-b68e-faf606561e34/home/neo/PycharmProjects/hed_origin/data/new_dataset/e_edge/augmentation/'

    orig_image_dst_folder_path = '../../data/new_dataset2/big/augmentation/'
    # orig_image_dst_folder_path = '/media/neo/92cd53f3-fc9f-4db8-b68e-faf606561e34/home/neo/PycharmProjects/hed_origin/data/new_dataset1/big/augmentation/'
    edge_image_dst_folder_path = '../../data/new_dataset2/e_edge/augmentation/'
    # edge_image_dst_folder_path = '/media/neo/92cd53f3-fc9f-4db8-b68e-faf606561e34/home/neo/PycharmProjects/hed_origin/data/new_dataset1/e_edge/augmentation/'

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


if __name__ == '__main__':
    choose_n_images(1000)
    # is_same()
