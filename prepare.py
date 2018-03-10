import json
import os
import time
from sys import getsizeof

import numpy as np
import tensorflow as tf

from constants import C
from utils import collect_annotations, create_full_mask, resize_mask


def create_mask_dirs(train, val):
    """
    Creates directories for masks. Requires paths
    """
    if not os.path.isdir(train):
        os.mkdir(train)
    if not os.path.isdir(val):
        os.mkdir(val)


def create_masks(annotations_path, dst_dir, dst_w, dst_h):
    """
    Creates masks for images. Masks are of the same size as images
    :param annotations_path: path to annotations file
    :param images: path to images dir
    :param dst_dir: destination to put masks
    :param dst_w: destination mask width
    :param dst_h: destination mask height
    """
    dataset = json.load(open(annotations_path, 'r'))
    annotations_list = dataset['annotations']
    images = dataset['images']

    mean_time = 0
    mean_size = 0
    data_len = len(images)
    for i, img_desc in enumerate(images):
        print('\rSaving masked labels: {} of {}, time left: {}, predicted size: {}'
              .format(i+1,
                      data_len,
                      calc_time_left(mean_time, data_len - i),
                      calc_pred_size(mean_size, data_len)),
              end='', flush=True)

        s = time.time()
        img_id = img_desc['id']
        img_w = img_desc['width']
        img_h = img_desc['height']
        img_annotations = collect_annotations(img_id, annotations_list)
        mask = create_full_mask(img_w, img_h, img_annotations)

        resized = resize_mask(mask, dst_w, dst_h)
        non_zero_ids = np.count_nonzero(resized, axis=(0, 1))
        non_zero_ids = np.nonzero(non_zero_ids)[0]
        resized = resized[:, :, non_zero_ids]

        mask_name = img_desc['file_name'].replace('jpg', 'npz')
        np.savez(os.path.join(dst_dir, mask_name), resized, non_zero_ids)

        e = time.time()
        mean_time = i / (i + 1) * mean_time + (e - s) / (i + 1)
        mean_size = i / (i + 1) * mean_size + resized.nbytes / (i + 1)


def calc_pred_size(mean_size, n_samples):
    """
    Calculates predicted size of dataset. Requires mean size of single sample in bytes
    """
    size = int(mean_size * n_samples / (2 ** 20))
    gigabytes = size // 1024
    megabytes = (size - gigabytes * 1024)
    return '{:d} GB {:d} MB'.format(gigabytes, megabytes)


def calc_time_left(mean_time, left_samples_n):
    """
    Computes formatted left time of any operation, given mean time of single operation and left number of samples
    """
    time_left = int(mean_time * left_samples_n)
    h = time_left // 3660
    m = (time_left - h * 3600) // 60
    s = (time_left - h * 3600 - m * 60)
    return '{:3d}h{:3d}m{:3d}s'.format(h, m, s)


def load_imgs_temp(dir):
    # todo delete or use in another place
    """
    Saved code to read label
    :param dir:
    :return:
    """
    files = os.listdir(dir)
    s = time.time()
    for i, filename in enumerate(files):
        print(i)
        arrays = np.load(os.path.join(dir, filename))
        slices = arrays['arr_0']
        ids = arrays['arr_1']
        label = np.zeros(shape=(dst_h, dst_w, C), dtype=np.uint8)
        label[:, :, ids] = slices
    e = time.time()
    print(e - s)


train_annotations = 'data/annotations/instances_train2017.json'
val_annotations = 'data/annotations/instances_val2017.json'
train_images = 'data/train2017'
val_images = 'data/val2017'
train_masks = 'data/train_masks'
val_masks = 'data/val_masks'

dst_w = 240
dst_h = 240

create_mask_dirs(train_masks, val_masks)
create_masks(val_annotations, val_masks, dst_w, dst_h)
