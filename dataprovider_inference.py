import os

import tensorflow as tf

from constants import img_w, img_h


class DataProvider:
    """
    Provides inference data for neural network
    """

    def __init__(self, filenames):
        """
        """
        self.filenames = filenames
        self.batch_size = 1
        self.img_w = img_w
        self.img_h = img_h

    def load_images(self, img_name):
        """
        Used within TF DatasetAPI, loads images, converts and scales them to range (0...1)
        """
        image_string = tf.read_file(img_name)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        image_resized = tf.image.resize_images(image_decoded, [self.img_h, self.img_w])
        image_resized /= 255
        b, g, r = tf.split(image_resized, 3, 2)
        return tf.concat([r, g, b], 2), img_name

    def get_data(self):
        """
        Creates dataset handles
        :param images_path:
        :return:
        """
        img_names = tf.constant(sorted(self.filenames))

        dataset = tf.data.Dataset.from_tensor_slices((img_names))
        dataset = dataset.map(self.load_images, num_parallel_calls=4)
        dataset = dataset.prefetch(self.batch_size)
        dataset = dataset.batch(self.batch_size)

        iterator = dataset.make_initializable_iterator()
        image, name = iterator.get_next()

        return image, name, iterator
