import numpy as np
import tensorflow as tf

from constants import C


def conv_decoder(encoder_output):
    """
    Creates a convolutional, few layer overlay on pretrained encoder
    """
    with tf.variable_scope('conv_decoder'):
        net = tf.layers.conv2d(encoder_output,
                               filters=256,
                               kernel_size=(1, 1),
                               padding='same',
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               activation=tf.nn.leaky_relu)
        net = tf.layers.conv2d(net,
                               filters=C,
                               kernel_size=(1, 1),
                               padding='same',
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               activation=None)
        return net
