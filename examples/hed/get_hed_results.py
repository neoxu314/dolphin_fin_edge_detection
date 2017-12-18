import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.cm as cm
import scipy.misc
from PIL import Image
import scipy.io
import os
import sys

# Make sure that caffe is on the python path:
caffe_root = '/opt/caffe/'  # this file is expected to be in {caffe_root}/examples/hed/
sys.path.insert(0, caffe_root + 'python')
import caffe


def get_image_list():
    basewidth = 300

    data_root = '../../data/HED-BSDS/'
    with open(data_root + 'segmentation_fin.lst') as f:
        test_lst = f.readlines()

    print("*******test_lst:**************", test_lst)

    test_lst = [data_root + x.strip() for x in test_lst]

    im_lst = []
    for i in range(0, len(test_lst)):
        # resize images
        im = Image.open(test_lst[i])

        im = im.convert('RGB')

        wpercent = (basewidth/float(im.size[0]))
        hsize = int((float(im.size[1]) * float(wpercent)))
        im = im.resize((basewidth, hsize), Image.ANTIALIAS)

        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:, :, ::-1]
        in_ -= np.array((104.00698793, 116.66876762, 122.67891434))
        im_lst.append(in_)

    return im_lst, test_lst


# Visualization
def get_hed_images(scale_lst, size, path):
    pylab.rcParams['figure.figsize'] = size, size / 2

    figure = plt.figure()
    for i in range(0, len(scale_lst)):
        s = plt.subplot(1, 5, i + 1)
        plt.imshow(1 - scale_lst[i], cmap=cm.Greys_r)
        s.set_xticklabels([])
        s.set_yticklabels([])
        s.yaxis.set_ticks_position('none')
        s.xaxis.set_ticks_position('none')

    figure.savefig(path)

    plt.tight_layout()


def get_img_save_path(name):
    root = '../../data/HED-BSDS/hed_segmentation_fins'
    return os.path.join(root, name)


def main():
    # remove the following two lines if testing with cpu
    caffe.set_mode_cpu()
    # caffe.set_device(0)

    # load net
    model_root = './'
    net = caffe.Net(model_root + 'deploy.prototxt', model_root + 'hed_pretrained_bsds.caffemodel', caffe.TEST)

    im_lst, name_lst = get_image_list()
    # for im, name in zip(im_lst, name_lst):
    #     print('image_name:', name)
    #     print('image_matrix', im)

    for img, name in zip(im_lst, name_lst):
        print("*********processing image: ", name)

        img_t = img.transpose((2, 0, 1))

        # shape for input (data blob is N x C x H x W), set data
        net.blobs['data'].reshape(1, *img_t.shape)
        net.blobs['data'].data[...] = img_t

        # run net and take argmax for prediction
        net.forward()
        # out1 = net.blobs['sigmoid-dsn1'].data[0][0, :, :]
        # out2 = net.blobs['sigmoid-dsn2'].data[0][0, :, :]
        # out3 = net.blobs['sigmoid-dsn3'].data[0][0, :, :]
        # out4 = net.blobs['sigmoid-dsn4'].data[0][0, :, :]
        # out5 = net.blobs['sigmoid-dsn5'].data[0][0, :, :]
        fuse = net.blobs['sigmoid-fuse'].data[0][0, :, :]

        scale_lst = [fuse]
        get_hed_images(scale_lst, 22, get_img_save_path(name.split('/')[-1]))
        # print(img_t)


if __name__ == '__main__':
    main()

