import os

import numpy as np
import tensorflow as tf

from constants import img_w, img_h, label_w_conv, label_h_conv, label_w_ae, label_h_ae, label_w_vae, label_h_vae, C


class DataProvider:
    """
    Provides data for neural network
    """

    def __init__(self, mode, batch_size):
        """
        Sets mode: conv, ae, vae - it changes image/label dimensiality
        :param mode: conv, ae or vae
        :param self.batch_size: size of batch
        """
        self.batch_size = batch_size
        self.img_w = img_w
        self.img_h = img_h

        if mode == 'conv_decoder':
            self.label_w = label_w_conv
            self.label_h = label_h_conv

        elif mode == 'ae_decoder':
            self.label_w = label_w_ae
            self.label_h = label_h_ae

        elif mode == 'vae_decoder':
            self.label_w = label_w_vae
            self.label_h = label_h_vae

    def dataset_resize_images(self, img_name, label_name):
        """
        Used within TF DatasetAPI, converts images and scales them to range (0...1)
        """
        image_string = tf.read_file(img_name)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        image_resized = tf.image.resize_images(image_decoded, [self.img_h, self.img_w])
        image_resized /= 255
        b, g, r = tf.split(image_resized, 3, 2)
        del image_string
        del image_resized
        return tf.concat([r, g, b], 2), label_name

    def dataset_convert_labels(self, img, label_name):
        arrays = np.load(label_name.decode())
        slices = arrays['arr_0']
        ids = arrays['arr_1']
        label = np.zeros(shape=(self.label_h, self.label_w, C), dtype=np.float32)
        label[:, :, ids] = slices
        del arrays
        del ids
        del slices
        return img, label

    def get_data(self, t_img_path, v_img_path, t_label_path, v_label_path):
        """
        Creates dataset init ops
        """
        train_label_names = tf.constant(sorted(os.path.join(t_label_path, name) for name in os.listdir(t_label_path)))
        val_label_names = tf.constant(sorted(os.path.join(v_label_path, name) for name in os.listdir(v_label_path)))
        train_image_names = tf.constant(sorted(os.path.join(t_img_path, name) for name in os.listdir(t_img_path)))
        val_image_names = tf.constant(sorted(os.path.join(v_img_path, name) for name in os.listdir(v_img_path)))

        training_dataset = tf.data.Dataset.from_tensor_slices((train_image_names, train_label_names))
        training_dataset = training_dataset.shuffle(buffer_size=50000)
        training_dataset = training_dataset.map(self.dataset_resize_images, num_parallel_calls=4)
        training_dataset = training_dataset.map(
            lambda filename, label: tuple(tf.py_func(self.dataset_convert_labels, [filename, label], [tf.float32, tf.float32], stateful=False)),
            num_parallel_calls=4)
        training_dataset = training_dataset.prefetch(self.batch_size)
        training_dataset = training_dataset.batch(self.batch_size)
        training_dataset = training_dataset.repeat()

        val_dataset = tf.data.Dataset.from_tensor_slices((val_image_names, val_label_names))
        val_dataset = val_dataset.shuffle(buffer_size=5000)
        val_dataset = val_dataset.map(self.dataset_resize_images, num_parallel_calls=4)
        val_dataset = val_dataset.map(
            lambda filename, label: tuple(tf.py_func(self.dataset_convert_labels, [filename, label], [tf.float32, tf.float32], stateful=False)),
            num_parallel_calls=4)
        val_dataset = val_dataset.prefetch(self.batch_size)
        val_dataset = val_dataset.batch(self.batch_size)
        val_dataset = val_dataset.repeat()

        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(handle, training_dataset.output_types, training_dataset.output_shapes)
        images, labels = iterator.get_next()

        training_iterator = training_dataset.make_one_shot_iterator()
        validation_iterator = val_dataset.make_one_shot_iterator()

        return handle, training_iterator, validation_iterator, images, labels
