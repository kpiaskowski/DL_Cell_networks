import json

import cv2
import numpy as np

C = 80
id_to_class = {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck',
               9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
               16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
               24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase',
               34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat',
               40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
               46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple',
               54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
               61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet',
               72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave',
               79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase',
               87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}
class_to_id = {'backpack': 27, 'spoon': 50, 'sports ball': 37, 'cat': 17, 'tie': 32, 'sheep': 20, 'elephant': 22,
               'toaster': 80, 'clock': 85, 'motorcycle': 4, 'broccoli': 56, 'skateboard': 41, 'bus': 6, 'laptop': 73,
               'horse': 19, 'person': 1, 'fire hydrant': 11, 'couch': 63, 'sandwich': 54, 'baseball bat': 39,
               'refrigerator': 82, 'carrot': 57, 'boat': 9, 'airplane': 5, 'skis': 35, 'giraffe': 25, 'dog': 18,
               'dining table': 67, 'surfboard': 42, 'frisbee': 34, 'bear': 23, 'hot dog': 58, 'baseball glove': 40,
               'toothbrush': 90, 'train': 7, 'bench': 15, 'snowboard': 36, 'mouse': 74, 'oven': 79, 'vase': 86,
               'bicycle': 2, 'kite': 38, 'cow': 21, 'wine glass': 46, 'cup': 47, 'traffic light': 10,
               'parking meter': 14, 'microwave': 78, 'umbrella': 28, 'zebra': 24, 'teddy bear': 88, 'cell phone': 77,
               'sink': 81, 'bottle': 44, 'apple': 53, 'truck': 8, 'chair': 62, 'scissors': 87, 'fork': 48,
               'handbag': 31, 'donut': 60, 'bird': 16, 'tv': 72, 'cake': 61, 'potted plant': 64, 'knife': 49,
               'hair drier': 89, 'book': 84, 'bowl': 51, 'remote': 75, 'bed': 65, 'suitcase': 33, 'stop sign': 13,
               'pizza': 59, 'car': 3, 'tennis racket': 43, 'toilet': 70, 'orange': 55, 'keyboard': 76, 'banana': 52}

# COCO classes aren't indexed from zero and there happen leaps between indices higher than 1
contid_to_COCOid = dict(zip(range(len(id_to_class.keys())), id_to_class.keys()))
COCOid_to_contid = dict(zip(id_to_class.keys(), range(len(id_to_class.keys()))))


def mask_image(org_w, org_h, annotations):
    """
    Creates tensor h x w x C with embedded binary masks. Preserves original image size. If there are multiple overlapping
    instances of objects of the same class, binary mask doesn't take it into account - it produces only one mask per
    class
    :param org_w: original image width
    :param org_h: original image heigth
    :param annotations: list of dicts - each item in dict should contain keys: 'category_id' -> int and 'bbox' -> [x, y, w, h] - other
    are unnecesary, but not prohibited
    :return: binary numpy array of shape w x h x C with ones, where objects are present
    """
    label = np.zeros([org_h, org_w, C], dtype=np.float32)
    for annotation in annotations:
        class_id = annotation['category_id']
        bbox = annotation['bbox']
        label[int(bbox[1]):int(bbox[1]) + int(bbox[3]), int(bbox[0]):int(bbox[0]) + int(bbox[2]),
        COCOid_to_contid[class_id]] = 1
    return label


def embedd_mask(img, mask):
    """
    Embedds binary tensor mask on image, using only one color. Resizes mask automatically in order to match img size.
    This function is rather for debugging than real usage
    :param img: real valued image (values from 0 to 1)
    :param mask: binary 3d mask
    :return: Img with embedded mask
    """
    img_w = img.shape[1]
    img_h = img.shape[0]
    label_w = mask.shape[1]
    label_h = mask.shape[0]

    w_step = img_w / label_w
    h_step = img_h / label_h

    flat_mask = np.max(mask, axis=2)
    img_size_mask = np.zeros_like(img)[..., 0]

    for y in range(label_h):
        for x in range(label_w):
            if flat_mask[y, x] == 1:
                img_size_mask[int(y * h_step):int((y + 1) * h_step), int(x * w_step):int((x + 1) * w_step)] = 1


                non_zero_column = np.count_nonzero(mask[int(y):int(y + 1), int(x):int(x + 1)], axis=(0, 1))
                present_classes = np.nonzero(non_zero_column)[0]

                if np.count_nonzero(non_zero_column) == 0:
                    print(np.count_nonzero(non_zero_column))

    # create green mask
    color_mask = np.stack([np.zeros_like(img_size_mask), img_size_mask, np.zeros_like(img_size_mask)], axis=2)
    return cv2.addWeighted(img, 0.7, color_mask, 0.3, 0)


def collect_annotations(image_id, annotations_list):
    """
    Collects all annotations for image, given its id
    :param annotations_list: list of all annotations
    :param image_id: integer
    :return: list of annotations for given image id
    """
    valid_annotations = [anno for anno in annotations_list if anno['image_id'] == image_id]
    return valid_annotations


def load_img(path):
    """
    Loads image and converts it to flaot32
    :param path:
    :return: real valued image of values between 0 and 1
    """
    img = cv2.imread(path)
    return (img / 255).astype(np.float32)


def resize_mask(org_mask, dst_w, dst_h):
    """
    Resizes binary mask to given shape according to width and height, leaves C unchanged. It actually creates tensor
    label for neural network
    :param org_mask: tensor of shape h x w x C
    :param dst_w: desired w, integer
    :param dst_h: desired h, integer
    :return: resized tensor (label) of shape dst_h x dst_w x c
    """
    w_step = mask.shape[1] / dst_w
    h_step = mask.shape[0] / dst_h
    fill_threshold = int(0.5 * int(w_step) * int(h_step))

    new_mask = np.zeros(shape=[dst_h, dst_w, C], dtype=np.float32)

    for y in range(dst_h):
        for x in range(dst_w):
            column = org_mask[int(y * h_step):int((y + 1) * h_step), int(x * w_step):int((x + 1) * w_step)]
            non_zero_column = np.count_nonzero(column, axis=(0, 1))
            non_zero_column[non_zero_column < fill_threshold] = 0
            non_zero_column[non_zero_column >= fill_threshold] = 1
            new_mask[y, x] = non_zero_column

    return new_mask


# todo embed text
# todo embed bounding box

annopath = 'data/annotations/instances_val2017.json'
dataset = json.load(open(annopath, 'r'))
annotations_list = dataset['annotations']
images_path = 'data/val2017'
images = dataset['images']

for i in range(20):
    # i = 1
    img_w = images[i]['width']
    img_h = images[i]['height']
    img_name = images[i]['file_name']
    img_id = images[i]['id']

    img = load_img(images_path + '/' + img_name)

    valid_annotations = collect_annotations(img_id, annotations_list)
    mask = mask_image(img_w, img_h, valid_annotations)

    new_mask = resize_mask(mask, 14,14)
    tagged2 = embedd_mask(np.copy(img), mask)
    tagged = embedd_mask(np.copy(img), new_mask)
    print(mask.shape, new_mask.shape)

    a = np.expand_dims(np.max(new_mask, axis=2), axis=2)

    cv2.imshow('org', tagged)
    cv2.imshow('org2', tagged2)
    cv2.imshow('res', a)
    cv2.waitKey(4000)
