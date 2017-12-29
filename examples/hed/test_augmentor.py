import os
import sys
sys.path.insert(0, os.path.abspath('.'))

import tempfile
import shutil

from PIL import Image
import Augmentor


def rotate_images(tmpdir, rot):

    original_dimensions = (800, 800)

    im_tmp = tmpdir.mkdir("subfolder").join('test.JPEG')
    im = Image.new('RGB', original_dimensions)
    im.save(str(im_tmp), 'JPEG')

    r = Augmentor.Operations.Rotate(probability=1, rotation=rot)
    im_r = r.perform_operation(im)

    assert im_r is not None
    assert im_r.size == original_dimensions


def test_rotate_images_90(tmpdir):
    rotate_images(tmpdir, 90)


test_rotate_images_90('../../data/augmentation_test_img')