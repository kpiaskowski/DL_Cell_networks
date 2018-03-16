import numpy as np
import tensorflow as tf

from constants import C


def conv_decoder(encoder_output):
    """
    Creates a convolutional, few layer overlay on pretrained encoder
    """
    namescope = 'conv_decoder'
    with tf.variable_scope(namescope):
        net = tf.layers.conv2d(encoder_output,
                               filters=256,
                               kernel_size=(1, 1),
                               padding='same',
                               kernel_initializer=tf.truncated_normal_initializer(0.0, 0.1),
                               activation=tf.nn.leaky_relu)
        net = tf.layers.conv2d(net,
                               filters=C,
                               kernel_size=(1, 1),
                               padding='same',
                               kernel_initializer=tf.truncated_normal_initializer(0.0, 0.1),
                               activation=None)
        return net, namescope
