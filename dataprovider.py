import os

import numpy as np
import tensorflow as tf

from constants import img_w_conv, img_h_conv, label_w_conv, label_h_conv, batch_size_conv, C

img_w = img_w_conv
img_h = img_h_conv
label_w = label_w_conv
label_h = label_h_conv
batch_size = batch_size_conv


def dataset_resize_images(img_name, label_name):
    """
    Used within TF DatasetAPI, converts images and scales them to range (0...1)
    """
    image_string = tf.read_file(img_name)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image_resized = tf.image.resize_images(image_decoded, [img_h, img_w])
    image_resized /= 255
    b, g, r = tf.split(image_resized, 3, 2)
    # del image_string
    # del image_resized
    return tf.concat([r, g, b], 2), label_name


def dataset_convert_labels(img, label_name):
    arrays = np.load(label_name.decode())
    slices = arrays['arr_0']
    ids = arrays['arr_1']
    label = np.zeros(shape=(label_h, label_w, C), dtype=np.float32)
    label[:, :, ids] = slices
    # del arrays
    # del ids
    # del slices
    return img, label


def datasets(t_img_path, v_img_path, t_label_path, v_label_path):
    """
    Creates dataset init ops
        """

    train_label_names = tf.constant(sorted(os.path.join(t_label_path, name) for name in os.listdir(t_label_path)))
    val_label_names = tf.constant(sorted(os.path.join(v_label_path, name) for name in os.listdir(v_label_path)))
    train_image_names = tf.constant(sorted(os.path.join(t_img_path, name) for name in os.listdir(t_img_path)))
    val_image_names = tf.constant(sorted(os.path.join(v_img_path, name) for name in os.listdir(v_img_path)))

    training_dataset = tf.data.Dataset.from_tensor_slices((train_image_names, train_label_names))
    training_dataset = training_dataset.shuffle(buffer_size=10000)
    training_dataset = training_dataset.map(dataset_resize_images, num_parallel_calls=4)
    training_dataset = training_dataset.map(
        lambda filename, label: tuple(tf.py_func(dataset_convert_labels, [filename, label], [tf.float32, tf.float32], stateful=False)),
        num_parallel_calls=4)
    training_dataset = training_dataset.prefetch(batch_size * 10)
    training_dataset = training_dataset.batch(batch_size)

    val_dataset = tf.data.Dataset.from_tensor_slices((val_image_names, val_label_names))
    val_dataset = val_dataset.shuffle(buffer_size=500)
    val_dataset = val_dataset.map(dataset_resize_images)
    val_dataset = val_dataset.map(
        lambda filename, label: tuple(tf.py_func(dataset_convert_labels, [filename, label], [tf.float32, tf.float32], stateful=False)))
    val_dataset = val_dataset.batch(batch_size)

    iterator = tf.data.Iterator.from_structure(training_dataset.output_types, training_dataset.output_shapes)
    img_input, labels = iterator.get_next()

    training_dataset_init = iterator.make_initializer(training_dataset)
    validation_dataset_init = iterator.make_initializer(val_dataset)

    return training_dataset_init, validation_dataset_init, img_input, labels
