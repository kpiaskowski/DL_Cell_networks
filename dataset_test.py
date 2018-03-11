import os

import cv2
import numpy as np
import tensorflow as tf

from utils import draw_predictions
from convolutional import convolutional_model
from constants import img_w, img_h, label_w, label_h, batch_size, C

label_names = tf.constant(sorted(os.path.join('data/labels_S14', name) for name in os.listdir('data/labels_S14')))
image_names = tf.constant(sorted(os.path.join('data/val2017', name) for name in os.listdir('data/val2017')))

def resize_images(img_name, label_name):
    image_string = tf.read_file(img_name)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image_resized = tf.image.resize_images(image_decoded, [img_h, img_w])
    image_resized /= 255
    b, g, r = tf.split(image_resized, 3, 2)
    return tf.concat([r, g, b], 2), label_name


def load_labels(img, label_name):
    arrays = np.load(label_name.decode())
    slices = arrays['arr_0']
    ids = arrays['arr_1']
    label = np.zeros(shape=(label_h, label_w, C), dtype=np.uint8)
    label[:, :, ids] = slices
    return img, label


training_dataset = tf.data.Dataset.from_tensor_slices((image_names, label_names))
training_dataset = training_dataset.map(resize_images)
training_dataset = training_dataset.map(lambda filename, label: tuple(tf.py_func(load_labels, [filename, label], [tf.float32, tf.uint8], stateful=False)))
training_dataset = training_dataset.shuffle(buffer_size=500)
training_dataset = training_dataset.batch(batch_size)


val_dataset = tf.data.Dataset.from_tensor_slices((image_names, label_names))
val_dataset = val_dataset.map(resize_images)
val_dataset = val_dataset.map(lambda filename, label: tuple(tf.py_func(load_labels, [filename, label], [tf.float32, tf.uint8], stateful=False)))
val_dataset = val_dataset.batch(batch_size)

iterator = tf.data.Iterator.from_structure(training_dataset.output_types, training_dataset.output_shapes)
img_input, label = iterator.get_next()

training_init_op = iterator.make_initializer(training_dataset)
validation_init_op = iterator.make_initializer(val_dataset)

model = convolutional_model(img_input)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(2):
        sess.run(training_init_op)
        for k in range(20):
            img, lbl = sess.run([img_input, label])
            out_model = sess.run(model)
            print(k, img.shape, out_model.shape)
            # cv2.imshow('', labelled_img)
            # cv2.waitKey(2000)

        # sess.run(validation_init_op)
        # img, lbl = sess.run([img_input, label])
        # labelled_img = draw_predictions(img, lbl, False, True)
        # print(img.dtype)
        # cv2.imshow('', labelled_img)
        # cv2.waitKey(2000)

# dataset = tf.data.Dataset.from_tensor_slices((image_names, label_names))
# dataset = dataset.map(resize_images)
# dataset = dataset.map(
#     lambda filename, label: tuple(tf.py_func(load_labels, [filename, label], [tf.uint8, tf.uint8], stateful=False)))
# iterator = dataset.make_one_shot_iterator()
# next_element = iterator.get_next()
#
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     out = sess.run(next_element)
#     print(out)

# todo clean up code
# todo zbuduj kod modelu i wykorzystaj te iteratory
# todo wygeneruj train set
# todo wydziel osobne listdiry do val i train, bo teraz jest wspolny

