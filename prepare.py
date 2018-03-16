import argparse
import json
import os
import time

import numpy as np

from utils import collect_annotations, create_full_mask, resize_mask


def create_mask_dirs(dir):
    """
    Creates directory for masks. Requires path
    """
    if not os.path.isdir(dir):
        os.mkdir(dir)


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
              .format(i + 1,
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


# parser = argparse.ArgumentParser(description='Prepares labels')
# parser.add_argument('--annotations_path ', help='path to FILE with annotations - that is the one with JSON extension', dest='annotations_path')
# parser.add_argument('--labels_path', help='path to folder, where labels should be placed - it will be created automatically if not present yet',
#                     dest='labels_path')
# parser.add_argument('--dst_w', help='width of generated labels', dest='dst_w')
# parser.add_argument('--dst_h', help='height of generated labels', dest='dst_h')
#
# args = parser.parse_args()
#
# annotations_path = args.annotations_path
# labels_path = args.labels_path
# dst_w = args.dst_w
# dst_h = args.dst_h
#
# if any((i is None for i in (annotations_path, labels_path, dst_h, dst_w))):
#     raise Exception('Provide all parameters (images_path, annotations_path, labels_path, dst_h, dst_w)!')
#
# create_mask_dirs(labels_path)
# create_masks(annotations_path, labels_path, int(dst_w), int(dst_h))

# quick help
# train_annotations = 'data/annotations/instances_train2017.json'
# val_annotations = 'data/annotations/instances_val2017.json'
