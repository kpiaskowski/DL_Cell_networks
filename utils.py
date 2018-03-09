import json

import cv2
import numpy as np
from constants import C, colors, id_to_class

# COCO classes aren't indexed from zero and there are leaps between indices higher than 1
contid_to_COCOid = dict(zip(range(len(id_to_class.keys())), id_to_class.keys()))
COCOid_to_contid = dict(zip(id_to_class.keys(), range(len(id_to_class.keys()))))


def create_full_mask(org_w, org_h, annotations):
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
        label[int(bbox[1]):int(bbox[1]) + int(bbox[3]), int(bbox[0]):int(bbox[0]) + int(bbox[2]), COCOid_to_contid[class_id]] = 1
    return label


def load_img(path):
    """
    Loads image and converts it to float32
    :param path:
    :return: real valued image of values between 0 and 1
    """
    img = cv2.imread(path)
    return (img / 255).astype(np.float32)


def collect_annotations(image_id, annotations_list):
    """
    Collects all annotations for image, given its id
    :param annotations_list: list of all annotations
    :param image_id: integer
    :return: list of annotations for given image id
    """
    valid_annotations = [anno for anno in annotations_list if anno['image_id'] == image_id]
    return valid_annotations


def resize_mask(org_mask, dst_w, dst_h):
    """
    Resizes binary mask to given shape according to width and height, leaves C unchanged. It actually creates tensor
    label for neural network
    :param org_mask: tensor of shape h x w x C
    :param dst_w: desired w, integer
    :param dst_h: desired h, integer
    :return: resized tensor (label) of shape dst_h x dst_w x c
    """
    planes = [cv2.resize(org_mask[:, :, i], dsize=(dst_w, dst_h)) for i in range(C)]
    stacked_planes = np.stack(planes, axis=-1)
    stacked_planes[stacked_planes > 0] = 1
    return stacked_planes


def compute_bboxes(mask):
    """
    Converts mask tensor into dict of bdboxes
    :param mask: tensor of predictions of shape
    :return: Dictionary of bboxes in range [0, C-1] and another dictionary converted to
    COCO classes space (they are not consecutively placed)
    """
    mask_planes = [mask[:, :, i] for i in range(C)]
    contours = [cv2.findContours(plane.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1] for plane in mask_planes]
    coords = [[cv2.boundingRect(cnt) for cnt in contour] for contour in contours]
    continuous_id_boxes = [[i, xywh] for i, xywh in enumerate(coords) if xywh]
    COCO_bboxes = [[contid_to_COCOid[i], xywh] for i, xywh in enumerate(coords) if xywh]
    return dict(continuous_id_boxes), dict(COCO_bboxes)


def draw_predictions(img, mask, show_mask=True, show_boxes=True):
    """
    Draws prediction on image. Resizes mask automatically in order to match img size.
    :param img: real valued image (values from 0 to 1)
    :param mask: binary 3d mask
    :param show_mask: True if mask should be drawn on image else False
    :param show_boxes: True if bounding boxes should be drawn
    :return: Img with drawn objects
    """

    img_w = img.shape[1]
    img_h = img.shape[0]
    label_w = mask.shape[1]
    label_h = mask.shape[0]
    w_step = img_w / label_w
    h_step = img_h / label_h

    new_img = np.copy(img)
    if show_mask:
        resized_mask = resize_mask(mask, img_w, img_h)
        colors_array = np.array(colors)
        color_mask = np.repeat(np.expand_dims(resized_mask, 3), 3, axis=-1)
        color_mask *= colors_array
        color_masks = [color_mask[:, :, i, :] for i in range(C)]
        output_color_mask = np.zeros_like(img)

        for plane in color_masks:
            output_color_mask = cv2.add(output_color_mask, plane)

        output_color_mask /= np.max(output_color_mask)
        new_img = cv2.addWeighted(new_img, 0.6, output_color_mask, 1, 0)

    if show_boxes:
        boxes, _ = compute_bboxes(mask)
        for k, v in boxes.items():
            for box in v:
                new_img = cv2.rectangle(new_img,
                                        (int(box[0] * w_step), int(box[1] * h_step)),
                                        (int((box[0] + box[2]) * w_step), int((box[1] + box[3]) * h_step)),
                                        color=colors[k],
                                        thickness=2)
                (text_w, text_h), _ = cv2.getTextSize(id_to_class[contid_to_COCOid[k]], cv2.FONT_HERSHEY_COMPLEX, 0.5, 1)
                new_img = cv2.rectangle(new_img,
                                        (int(box[0] * w_step), int(box[1] * h_step)),
                                        (int(box[0] * w_step) + text_w + 5, int(box[1] * h_step) + text_h + 5),
                                        color=(1, 1, 1),
                                        thickness=-1)
                new_img = cv2.putText(new_img,
                                      id_to_class[contid_to_COCOid[k]],
                                      (int(box[0] * w_step), int(box[1] * h_step) + 10),
                                      cv2.FONT_HERSHEY_COMPLEX,
                                      fontScale=0.5,
                                      color=(0, 0, 0))

    return new_img
