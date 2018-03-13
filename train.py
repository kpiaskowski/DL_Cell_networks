import os

import cv2
import numpy as np
import tensorflow as tf

from constants import img_w_conv, img_h_conv, label_w_conv, label_h_conv, batch_size_conv, C
from convolutional import convolutional_model
from utils import create_training_dirs, draw_predictions

# experiment params
epochs = 10
l_rate = 0.0001
thresh = 0.5
model_name = 'test5'

img_w = img_w_conv
img_h = img_h_conv
label_w = label_w_conv
label_h = label_h_conv
batch_size = batch_size_conv
architecture = convolutional_model

t_img_path = 'data/val2017'
v_img_path = 'data/val2017'

t_label_path = 'data/val_labels_S14'
v_label_path = 'data/val_labels_S14'

# data
t_num_batches = len(os.listdir(t_label_path)) // batch_size
v_num_batches = 5  # len(os.listdir(v_label_path)) // batch_size

train_label_names = tf.constant(sorted(os.path.join(t_label_path, name) for name in os.listdir(t_label_path)))
val_label_names = tf.constant(sorted(os.path.join(v_label_path, name) for name in os.listdir(v_label_path)))
train_image_names = tf.constant(sorted(os.path.join(t_img_path, name) for name in os.listdir(t_img_path)))
val_image_names = tf.constant(sorted(os.path.join(v_img_path, name) for name in os.listdir(v_img_path)))


def dataset_resize_images(img_name, label_name):
    """
    Used within TF DatasetAPI, converts images and scales them to range (0...1)
    """
    image_string = tf.read_file(img_name)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image_resized = tf.image.resize_images(image_decoded, [img_h, img_w])
    image_resized /= 255
    b, g, r = tf.split(image_resized, 3, 2)
    return tf.concat([r, g, b], 2), label_name


def dataset_convert_labels(img, label_name):
    arrays = np.load(label_name.decode())
    slices = arrays['arr_0']
    ids = arrays['arr_1']
    label = np.zeros(shape=(label_h, label_w, C), dtype=np.uint8)
    label[:, :, ids] = slices
    return img, label


training_dataset = tf.data.Dataset.from_tensor_slices((train_image_names, train_label_names))
training_dataset = training_dataset.map(dataset_resize_images)
training_dataset = training_dataset.map(
    lambda filename, label: tuple(tf.py_func(dataset_convert_labels, [filename, label], [tf.float32, tf.uint8], stateful=False)))
training_dataset = training_dataset.shuffle(buffer_size=500)
training_dataset = training_dataset.batch(batch_size)

val_dataset = tf.data.Dataset.from_tensor_slices((val_image_names, val_label_names))
val_dataset = val_dataset.map(dataset_resize_images)
val_dataset = val_dataset.map(lambda filename, label: tuple(tf.py_func(dataset_convert_labels, [filename, label], [tf.float32, tf.uint8], stateful=False)))
val_dataset = val_dataset.batch(batch_size)

iterator = tf.data.Iterator.from_structure(training_dataset.output_types, training_dataset.output_shapes)
img_input, labels = iterator.get_next()

training_dataset_init = iterator.make_initializer(training_dataset)
validation_dataset_init = iterator.make_initializer(val_dataset)

logits = architecture(img_input)

output = tf.nn.sigmoid(logits)
loss = tf.losses.sigmoid_cross_entropy(labels, logits)
train_op = tf.train.AdamOptimizer(l_rate).minimize(loss)

# saving and logging
create_training_dirs('saved_models', 'saved_summaries', 'generated_images', model_name)
saver = tf.train.Saver()
with tf.name_scope('summaries'):
    tf.summary.scalar('loss', loss)
merged = tf.summary.merge_all()

with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(os.path.join('saved_summaries', model_name, 'train'), sess.graph)
    val_writer = tf.summary.FileWriter(os.path.join('saved_summaries', model_name, 'validation'))
    sess.run(tf.global_variables_initializer())
    if len(os.listdir(os.path.join('saved_models', model_name))) > 0:
        saver.restore(sess, os.path.join('saved_models', model_name, 'model.ckpt'))
        print(model_name + ' loaded')
    else:
        print('Training %s from scratch' % model_name)

    for epoch in range(epochs):
        sess.run(training_dataset_init)
        for k in range(t_num_batches):
            _, cost, summary = sess.run([train_op, loss, merged])
            print("\rTraining, epoch: {} of {}, batch: {} of {}, cost: {}".format(epoch + 1, epochs, k + 1, t_num_batches, cost), end='', flush=True)
            train_writer.add_summary(summary, epoch * t_num_batches + k)
        saver.save(sess, os.path.join('saved_models', model_name, 'model.ckpt'))
        print()

        sess.run(validation_dataset_init)
        for k in range(v_num_batches):
            cost, summary = sess.run([loss, merged])
            print("\rTesting, epoch: {} of {}, batch: {} of {}, cost: {}".format(epoch + 1, epochs, k + 1, v_num_batches, cost), end='', flush=True)
            val_writer.add_summary(summary, epoch * v_num_batches + k)
        masks, imgs = sess.run([output, img_input])
        print()

        for i in range(batch_size):
            mask = masks[i]
            img = imgs[i]
            mask[mask >= thresh] = 1
            mask[mask < thresh] = 0
            labelled_img = draw_predictions(img, mask)
            cv2.imwrite(os.path.join('generated_images', model_name, 'e_' + str(epoch+1) + '_i_' + str(i) + '.jpg'), labelled_img * 255)
