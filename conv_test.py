import json
import os

import numpy as np
import tensorflow as tf
from architectures.conv_decoder import conv_decoder
from architectures.pretrained_encoder import pretrained_encoder as encoder

from constants import label_w_conv as label_size
from dataprovider_inference import DataProvider
from utils import draw_predictions, valid_filenames, compute_coco_annotations, contid_to_COCOid

# inference params
model_name = 'model_s14'
model_checkpoint = '62695'
data_path = 'data/test2017'
annotations_path = 'data/test_annotations/image_info_test-dev2017.json'
thresh = 0.3

name_dict, filenames = valid_filenames(annotations_path, data_path)

# datasets
dataset = DataProvider(filenames)
images, names = dataset.get_data()

# model
encoder = encoder(images)
logits = conv_decoder(encoder)
output = tf.nn.sigmoid(logits)

# loader
loader = tf.train.Saver()

results = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    loader.restore(sess, os.path.join('saved_models', model_name, 'model.ckpt-' + model_checkpoint))

    i = 0
    while True:
        try:
            predictions, image, name = sess.run([output, images, names])  # since we are using only batchsize = 1
            id, w, h = name_dict[name[0].decode()]['id'], name_dict[name[0].decode()]['height'], name_dict[name[0].decode()]['width']

            mask = np.copy(predictions)
            mask[mask >= thresh] = 1
            mask[mask < thresh] = 0
            labelled_img, boxes = draw_predictions(image, mask, False, False)
            scores = compute_coco_annotations(boxes, predictions[0], w, h, contid_to_COCOid, label_size)
            for k, boxes in scores.items():
                for box in boxes:
                    results.append({
                        'image_id': id,
                        'category_id': k,
                        'bbox': box[:4],
                        'score': box[-1]
                    })
            # cv2.imshow('', labelled_img)
            # cv2.waitKey(2000)
            print('Processing image {} of {}'.format(i + 1, len(filenames)))
            i += 1
        except tf.errors.OutOfRangeError:
            print("End of test set!")
            break

if not os.path.isdir('results'):
    os.mkdir('results')

with open('results/' + 'detections_test_dev2017_cellnets-' + model_name + '_results.json', 'w') as outfile:
    json.dump(results, outfile)
