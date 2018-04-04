import argparse
import os

import cv2
import numpy as np
import tensorflow as tf
from architectures.conv_decoder import conv_decoder
from architectures.pretrained_encoder import pretrained_encoder as encoder

from dataprovider_inference import DataProvider
from utils import draw_predictions

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predicts bounding boxes for sample images')
    parser.add_argument('--model_name', help='name of model', dest='model_name')
    parser.add_argument('--model_checkpoint', help='number of checkpoint', dest='model_checkpoint')
    parser.add_argument('--thresh', help='threshold, default value = 0.3', dest='thresh')
    args = parser.parse_args()

    model_name = args.model_name
    model_checkpoint = args.model_checkpoint
    thresh = float(args.thresh)

    # datasets
    filenames = ['sample_images/' + name for name in os.listdir('sample_images')]
    dataset = DataProvider(filenames)
    images, names, iterator = dataset.get_data()

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
        sess.run(iterator.initializer)
        while True:
            try:
                predictions, image, name = sess.run([output, images, names])  # since we are using only batchsize = 1
                mask = np.copy(predictions)
                mask[mask >= thresh] = 1
                mask[mask < thresh] = 0

                labelled_img, boxes = draw_predictions(image, mask, False, True)
                print(np.max(labelled_img), np.min(labelled_img))
                cv2.imwrite(name[0].decode(), labelled_img * 255)

            except tf.errors.OutOfRangeError:
                print("End of sample images!")
                break
