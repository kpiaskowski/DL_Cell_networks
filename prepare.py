import json
import cv2
import numpy as np
from utils import load_img, collect_annotations, create_full_mask, resize_mask, draw_predictions

annopath = 'data/annotations/instances_val2017.json'
dataset = json.load(open(annopath, 'r'))
annotations_list = dataset['annotations']
images_path = 'data/val2017'
images = dataset['images']

i = 7
img_w = images[i]['width']
img_h = images[i]['height']
img_name = images[i]['file_name']
img_id = images[i]['id']

img = load_img(images_path + '/' + img_name)
valid_annotations = collect_annotations(img_id, annotations_list)
mask = create_full_mask(img_w, img_h, valid_annotations)

resized = resize_mask(mask, 14, 14)
resized_mask_img = np.max(resized, axis=-1)
# mas2 = draw_predictions(img, mask, colors)

# cv2.imshow('new', resized_mask_img)
mas3 = draw_predictions(img, resized, show_mask=False)
cv2.imshow('mas_unresized', mas3)
cv2.waitKey(-1)