import os
import time
import xml.etree.ElementTree as ET
import cv2
import numpy as np

from prepare import create_mask_dirs, calc_time_left, calc_pred_size
from constants import colors

C = 15
dst_w = 14
dst_h = 14
labels_path = 'imagenet_data/labels'
annotations_path = 'imagenet_data/annotations'
images_path = 'imagenet_data/images'

classes = ['button', 'wrench', 'pliers', 'scissors', 'vial', 'screwdriver', 'tape', 'hammer', 'bottle', 'light_bulb', 'nail', 'screw', 'driller', 'broom',
           'axe']
colors = colors[:15]

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
    return dict(continuous_id_boxes)

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
        if np.max(output_color_mask) != 0:
            output_color_mask /= np.max(output_color_mask)

        new_img = cv2.addWeighted(new_img, 0.6, output_color_mask, 1, 0)

    if show_boxes:
        boxes = compute_bboxes(mask)
        for k, v in boxes.items():
            for box in v:
                new_img = cv2.rectangle(new_img,
                                        (int(box[0] * w_step), int(box[1] * h_step)),
                                        (int((box[0] + box[2]) * w_step), int((box[1] + box[3]) * h_step)),
                                        color=colors[k],
                                        thickness=2)
                (text_w, text_h), _ = cv2.getTextSize(classes[k], cv2.FONT_HERSHEY_COMPLEX, 0.5, 1)
                new_img = cv2.rectangle(new_img,
                                        (int(box[0] * w_step), int(box[1] * h_step)),
                                        (int(box[0] * w_step) + text_w + 5, int(box[1] * h_step) + text_h + 5),
                                        color=(1, 1, 1),
                                        thickness=-1)
                new_img = cv2.putText(new_img,
                                      classes[k],
                                      (int(box[0] * w_step), int(box[1] * h_step) + 10),
                                      cv2.FONT_HERSHEY_COMPLEX,
                                      fontScale=0.5,
                                      color=(0, 0, 0))

    return new_img


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
    label = np.zeros([org_h, org_w, C], dtype=np.uint8)
    for annotation in annotations:
        class_id = annotation['category_id']
        bbox = annotation['bbox']
        label[int(bbox[1]):int(bbox[1]) + int(bbox[3]), int(bbox[0]):int(bbox[0]) + int(bbox[2]), class_id] = 1
    return label


def create_masks(annotations_path, dst_dir, dst_w, dst_h):
    """
    Creates masks for images. Masks are of the same size as images
    :param annotations_path: path to annotations file
    :param images: path to images dir
    :param dst_dir: destination to put masks
    :param dst_w: destination mask width
    :param dst_h: destination mask height
    """
    images = [os.path.join(images_path, name) for name in os.listdir(images_path)]
    xmls = [os.path.join(annotations_path, name) for name in os.listdir(annotations_path)]


    mean_time = 0
    mean_size = 0
    data_len = len(images)
    for i, xml in enumerate(xmls):
        print('\rSaving masked labels: {} of {}, time left: {}, predicted size: {}'
              .format(i + 1,
                      data_len,
                      calc_time_left(mean_time, data_len - i),
                      calc_pred_size(mean_size, data_len)),
              end='', flush=True)

        s = time.time()

        # read xml
        tree = ET.parse(xml)
        size = tree.find('size')
        img_w = int(size.find('width').text)

        img_h = int(size.find('height').text)
        img_annotations = []
        objs = tree.findall('object')

        for obj in objs:
            bbox = obj.find('bndbox')
            x = float(bbox.find('xmin').text)
            y = float(bbox.find('ymin').text)
            w = float(bbox.find('xmax').text) - x
            h = float(bbox.find('ymax').text) - y
            class_id = classes.index(obj.find('name').text)
            img_annotations.append({'category_id': class_id, 'bbox': (x, y, w, h)})

        mask = create_full_mask(img_w, img_h, img_annotations)

        resized = resize_mask(mask, dst_w, dst_h)
        non_zero_ids = np.count_nonzero(resized, axis=(0, 1))
        non_zero_ids = np.nonzero(non_zero_ids)[0]
        resized = resized[:, :, non_zero_ids]

        mask_name = xml.replace('xml', 'npz').replace(annotations_path, dst_dir)

        np.savez(mask_name, resized, non_zero_ids)

        e = time.time()
        mean_time = i / (i + 1) * mean_time + (e - s) / (i + 1)
        mean_size = i / (i + 1) * mean_size + resized.nbytes / (i + 1)

def draw_predicted_cells(float_img, thresh_logits, S, src_img_size):
    step = int(src_img_size / S)
    overlay = np.max(thresh_logits, axis=2)
    output = np.ones_like(float_img) * 0.5
    for y in range(S):
        for x in range(S):
            x_s = int(x * src_img_size / S)
            x_e = int((x + 1) * src_img_size / S)
            y_s = int(y * src_img_size / S)
            y_e = int((y + 1) * src_img_size / S)
            output[y_s: y_e, x_s: x_e] *= overlay[y, x]

    output = cv2.addWeighted(float_img, 1, output, 0.4, 0)
    for y in range(S):
        for x in range(S):
            for c in range(thresh_logits.shape[-1]):
                if thresh_logits[y, x, c] == 1:
                    output = cv2.putText(output, str(c), (x * step + 10, y * step + 20), cv2.FONT_HERSHEY_COMPLEX, 0.6,
                                         (0, 0, 0), 1)
    return output


def draw_bounding_boxes(float_img, bounding_boxes, colours):
    """
    Draws bounding boxes on image
    :param float_img: real valued image
    :param bounding_boxes: list of bounding boxes, each [class, x1, y1, x2, y2, confidence]
    :param colours: list of colours per class
    :return: image with embedded bounding boxes
    """
    for bdbox in bounding_boxes:
        text_size, _ = cv2.getTextSize('%s %.2f' % (classes[bdbox[0]], bdbox[-1]), cv2.FONT_HERSHEY_COMPLEX, 0.5, 1)
        float_img = cv2.rectangle(float_img, (bdbox[1], bdbox[2]), (bdbox[1] + text_size[0] + 5, bdbox[2] + 20),
                                  colours[bdbox[0]], thickness=-1)
        float_img = cv2.rectangle(float_img, (bdbox[1], bdbox[2]), (bdbox[3], bdbox[4]), colours[bdbox[0]], thickness=2)
        float_img = cv2.putText(float_img, '%s %.2f' % (classes[bdbox[0]], bdbox[-1]), (bdbox[1] + 5, bdbox[2] + 15),
                                cv2.FONT_HERSHEY_COMPLEX, 0.5,
                                (0, 0, 0) if np.sum(colours[bdbox[0]]) > 1 else (1, 1, 1), thickness=1)
    return float_img


def get_bounding_boxes(thresh_logits, logits, S, src_img_size, min_contour_area=2.0):
    """
    Creates list of bounding boxes
    :param logits: real valued logits (needed to compute confidence
    :param thresh_logits: binary logits (0 or 1)
    :param S: number of cells per side
    :param src_img_size: size of image
    :return: list of bounding boxes, each box: [x1, y1, x2, y2, class]
    """
    step = int(src_img_size / S)

    # find contours and appropriate bounding boxes
    bounding_boxes = []
    for class_id in range(thresh_logits.shape[-1]):
        plane = np.copy(thresh_logits[..., class_id:class_id + 1]).astype(np.uint8)
        _, contours, _ = cv2.findContours(plane, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if cv2.contourArea(contour) <= min_contour_area:
                    continue
                confidence = np.sum(logits[y:y + h, x: x + w, class_id]) / np.count_nonzero(
                    logits[y:y + h, x: x + w, class_id])
                bounding_boxes.append([class_id, x * step, y * step, (x + w) * step, (y + h) * step, confidence])
    return bounding_boxes


# create_mask_dirs(labels_path)
# create_masks(annotations_path, labels_path, int(dst_w), int(dst_h))
