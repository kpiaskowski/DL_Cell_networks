import os

import cv2
import numpy as np
import tensorflow as tf

from architectures.conv_decoder import conv_decoder
from architectures.pretrained_encoder import pretrained_encoder as encoder
from constants import img_w_conv, img_h_conv, label_w_conv, label_h_conv, batch_size_conv, C
from dataprovider import datasets
from utils import create_training_dirs, draw_predictions

# experiment params
epochs = 10
l_rate = 0.00001
thresh = 0.5
model_name = 'test'
decoder = conv_decoder
decoder_namespace = 'conv_decoder'
batch_size = batch_size_conv

t_img_path = 'data/val2017'
v_img_path = 'data/val2017'

t_label_path = 'data/val_labels_S14'
v_label_path = 'data/val_labels_S14'

# data
t_num_batches = len(os.listdir(t_label_path)) // batch_size
v_num_batches = 5  # len(os.listdir(v_label_path)) // batch_size
training_dataset_init, validation_dataset_init, img_input, labels = datasets(t_img_path, v_img_path, t_label_path, v_label_path)

encoder = encoder(img_input)
logits = decoder(encoder)

output = tf.nn.sigmoid(logits)
loss = tf.losses.sigmoid_cross_entropy(labels, logits)
train_op = tf.train.AdamOptimizer(l_rate).minimize(loss)

# saving and logging
create_training_dirs('saved_models', 'saved_summaries', 'generated_images', model_name)
saver_decoder = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=decoder_namespace))
saver_encoder = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='yolo'))

with tf.name_scope('summaries'):
    tf.summary.scalar('loss', loss)
merged = tf.summary.merge_all()

with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(os.path.join('saved_summaries', model_name, 'train'), sess.graph)
    val_writer = tf.summary.FileWriter(os.path.join('saved_summaries', model_name, 'validation'))
    sess.run(tf.global_variables_initializer())
    saver_encoder.restore(sess, 'pretrained_imagenet/pretrained_imagenet.ckpt')  # load pretrained encoder

    if len(os.listdir(os.path.join('saved_models', model_name))) > 0:
        saver_decoder.restore(sess, os.path.join('saved_models', model_name, 'model.ckpt'))
        print(model_name + ' loaded')
    else:
        print('Training %s from scratch' % model_name)

    for epoch in range(epochs):
        sess.run(training_dataset_init)
        for k in range(t_num_batches):
            _, cost, summary = sess.run([train_op, loss, merged])
            print("\rTraining, epoch: {} of {}, batch: {} of {}, cost: {}".format(epoch + 1, epochs, k + 1, t_num_batches, cost), end='', flush=True)
            train_writer.add_summary(summary, epoch * t_num_batches + k)
        saver_decoder.save(sess, os.path.join('saved_models', model_name, 'model.ckpt'))
        print()

        sess.run(validation_dataset_init)
        for k in range(v_num_batches):
            cost, summary = sess.run([loss, merged])
            print("\rTesting, epoch: {} of {}, batch: {} of {}, cost: {}".format(epoch + 1, epochs, k + 1, v_num_batches, cost), end='', flush=True)
            val_writer.add_summary(summary, epoch * v_num_batches + k)
        masks, imgs = sess.run([output, img_input])
        print()

        for i in range(3):
            mask = masks[i]
            img = imgs[i]
            print(np.max(mask))
            mask[mask >= thresh] = 1
            mask[mask < thresh] = 0
            labelled_img = draw_predictions(img, mask)
            cv2.imwrite(os.path.join('generated_images', model_name, 'e_' + str(epoch + 1) + '_i_' + str(i) + '.jpg'), labelled_img * 255)
