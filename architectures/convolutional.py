"""
Convolutional model (not autoencoder), which expects S to be rather small, like for eg. S=14. Based on original YOLO. To change final output shape,
you have to modify network layers.
"""

import numpy as np
import tensorflow as tf

from constants import img_h_conv, img_w_conv, C


def convolutional_model(input_placeholder):
    """Returns convolutional model"""

    with tf.variable_scope('convolutional'):
        net = tf.reshape(input_placeholder, shape=[-1, img_h_conv, img_w_conv, 3])  # hack to make TF work - DatasetAPI somehow doesn't pass number of channels :/
        net = tf.pad(net, np.array([[0, 0], [3, 3], [3, 3], [0, 0]]))
        net = tf.layers.conv2d(net,
                               filters=64,
                               kernel_size=(7, 7),
                               strides=(2, 2),
                               padding='VALID',
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               activation=tf.nn.leaky_relu)
        net = tf.layers.max_pooling2d(net,
                                      pool_size=(2, 2),
                                      strides=(2, 2),
                                      padding='SAME')
        net = tf.layers.conv2d(net,
                               filters=192,
                               kernel_size=(3, 3),
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               activation=tf.nn.leaky_relu)
        net = tf.layers.max_pooling2d(net,
                                      pool_size=(2, 2),
                                      strides=(2, 2),
                                      padding='SAME')
        net = tf.layers.conv2d(net,
                               filters=128,
                               kernel_size=(1, 1),
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               activation=tf.nn.leaky_relu)
        net = tf.layers.conv2d(net,
                               filters=256,
                               kernel_size=(3, 3),
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               activation=tf.nn.leaky_relu)
        net = tf.layers.conv2d(net,
                               filters=256,
                               kernel_size=(1, 1),
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               activation=tf.nn.leaky_relu)
        net = tf.layers.conv2d(net,
                               filters=512,
                               kernel_size=(3, 3),
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               activation=tf.nn.leaky_relu)
        net = tf.layers.max_pooling2d(net,
                                      pool_size=(2, 2),
                                      strides=(2, 2),
                                      padding='SAME')
        net = tf.layers.conv2d(net,
                               filters=256,
                               kernel_size=(1, 1),
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               activation=tf.nn.leaky_relu)
        net = tf.layers.conv2d(net,
                               filters=512,
                               kernel_size=(3, 3),
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               activation=tf.nn.leaky_relu)
        net = tf.layers.conv2d(net,
                               filters=256,
                               kernel_size=(1, 1),
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               activation=tf.nn.leaky_relu)
        net = tf.layers.conv2d(net,
                               filters=512,
                               kernel_size=(3, 3),
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               activation=tf.nn.leaky_relu)
        net = tf.layers.conv2d(net,
                               filters=256,
                               kernel_size=(1, 1),
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               activation=tf.nn.leaky_relu)
        net = tf.layers.conv2d(net,
                               filters=512,
                               kernel_size=(3, 3),
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               activation=tf.nn.leaky_relu)
        net = tf.layers.conv2d(net,
                               filters=256,
                               kernel_size=(1, 1),
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               activation=tf.nn.leaky_relu)
        net = tf.layers.conv2d(net,
                               filters=512,
                               kernel_size=(3, 3),
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               activation=tf.nn.leaky_relu)
        net = tf.layers.conv2d(net,
                               filters=512,
                               kernel_size=(1, 1),
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               activation=tf.nn.leaky_relu)
        net = tf.layers.conv2d(net,
                               filters=1024,
                               kernel_size=(3, 3),
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               activation=tf.nn.leaky_relu)
        net = tf.layers.max_pooling2d(net,
                                      pool_size=(2, 2),
                                      strides=(1, 1),
                                      padding='SAME')
        net = tf.layers.conv2d(net,
                               filters=512,
                               kernel_size=(1, 1),
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               activation=tf.nn.leaky_relu)
        # (batch_size, 18, 18, 1024)
        net = tf.layers.conv2d(net,
                               filters=1024,
                               kernel_size=(3, 3),
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               activation=tf.nn.leaky_relu)
        # (batch_size, 16, 16, 512)
        net = tf.layers.conv2d(net,
                               filters=512,
                               padding='SAME',
                               kernel_size=(3, 3),
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               activation=tf.nn.leaky_relu)
        # (batch_size, 14, 14, 512)
        net = tf.layers.conv2d(net,
                               filters=512,
                               kernel_size=(1, 1),
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               activation=tf.nn.leaky_relu)
        # (batch_size, 14, 14, 256)
        net = tf.layers.conv2d(net,
                               filters=256,
                               kernel_size=(1, 1),
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               activation=tf.nn.leaky_relu)
        # (batch_size, 14, 14, 128)
        net = tf.layers.conv2d(net,
                               filters=C,
                               kernel_size=(1, 1),
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               activation=tf.nn.leaky_relu)
        return net
