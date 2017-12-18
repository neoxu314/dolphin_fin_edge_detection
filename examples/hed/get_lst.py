import os
import re


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


def main():
    root_directory = '../../data/HED-BSDS/Final_Database'
    list_path = '../../data/HED-BSDS/segmentation_fin.lst'
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
        path = os.path.join('../../data/HED-BSDS/Final_Database', folder_name)
        # path = os.path.join(path, '/PNG')
        path = path + '/PNG'
        sub_directories.append(path)

    print(sub_directories)

    for directory in sub_directories:
        image_paths.extend(get_image_paths(directory))

    print('*********************image paths**********************')
    print(image_paths)

    list_file = open(list_path, 'w')
    for path in image_paths:
        npath = path[path.find('Final_Database'):]
        list_file.write("%s\n" % npath)


if __name__ == '__main__':
    main()