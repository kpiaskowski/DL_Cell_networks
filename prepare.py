import argparse
import json
import multiprocessing as mp
import os
import time
from multiprocessing import Value

import numpy as np

from utils import collect_annotations, create_full_mask, resize_mask, create_mask_dirs, calc_time_left, calc_pred_size


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

    num_processes = mp.cpu_count()
    data_len = len(images)
    divider = data_len / num_processes

    images = [images[int(i * divider): int((i + 1) * divider)] for i in range(num_processes)]

    step = Value('i', 0)
    mean_time = Value('f', 0)
    mean_size = Value('f', 0)

    def process_labels(images):
        for i, img_desc in enumerate(images):
            step.value += 1
            print('\rSaving masked labels: {} of {}, time left: {}, predicted size: {}'
                  .format(step.value,
                          data_len,
                          calc_time_left(mean_time, data_len - step.value),
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
            mean_time.value = i / (i + 1) * mean_time.value + (e - s) / (i + 1) / num_processes
            mean_size.value = i / (i + 1) * mean_size.value + resized.nbytes / (i + 1)

    processes = [mp.Process(target=process_labels, args=([images[i]])) for i in range(num_processes)]
    for p in processes:
        p.start()

    for p in processes:
        p.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepares labels')
    parser.add_argument('--annotations_path ', help='path to FILE with annotations - that is the one with JSON extension', dest='annotations_path')
    parser.add_argument('--labels_path', help='path to folder, where labels should be placed - it will be created automatically if not present yet',
                        dest='labels_path')
    parser.add_argument('--dst_w', help='width of generated labels', dest='dst_w')
    parser.add_argument('--dst_h', help='height of generated labels', dest='dst_h')

    args = parser.parse_args()

    annotations_path = args.annotations_path
    labels_path = args.labels_path
    dst_w = args.dst_w
    dst_h = args.dst_h

    if any((i is None for i in (annotations_path, labels_path, dst_h, dst_w))):
        raise Exception('Provide all parameters (images_path, annotations_path, labels_path, dst_h, dst_w)!')

    create_mask_dirs(labels_path)
    create_masks(annotations_path, labels_path, int(dst_w), int(dst_h))

# quick path help
# train_annotations = 'data/annotations/instances_train2017.json'
# val_annotations = 'data/annotations/instances_val2017.json'
