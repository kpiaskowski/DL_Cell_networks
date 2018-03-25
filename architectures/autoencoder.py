import numpy as np
import tensorflow as tf

from constants import C


def ae_decoder(encoder_output):
    """
    Creates a decoder (autoencoder style)
    """
    namescope = 'ae_decoder'
    with tf.variable_scope(namescopem):
        # encoder continuation
        net = tf.layers.conv2d(encoder_output,
                               filters=256,
                               kernel_size=(1, 1),
                               padding='valid',
                               kernel_initializer=tf.truncated_normal_initializer(0.0, 0.1),
                               activation=tf.nn.elu)
        net = tf.layers.conv2d(net,
                               filters=64,
                               kernel_size=(3, 3),
                               padding='valid',
                               kernel_initializer=tf.truncated_normal_initializer(0.0, 0.1),
                               activation=tf.nn.elu)
        # decoder
        net = tf.image.resize_images(net, size=(24, 24), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        net = tf.layers.conv2d(net,
                               filters=256,
                               kernel_size=(3, 3),
                               padding='same',
                               kernel_initializer=tf.truncated_normal_initializer(0.0, 0.1),
                               activation=tf.nn.elu)
        net = tf.image.resize_images(net, size=(48, 48), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        net = tf.layers.conv2d(net,
                               filters=256,
                               kernel_size=(3, 3),
                               padding='same',
                               kernel_initializer=tf.truncated_normal_initializer(0.0, 0.1),
                               activation=tf.nn.elu)
        net = tf.image.resize_images(net, size=(128, 128), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        net = tf.layers.conv2d(net,
                               filters=256,
                               kernel_size=(3, 3),
                               padding='same',
                               kernel_initializer=tf.truncated_normal_initializer(0.0, 0.1),
                               activation=tf.nn.elu)
        net = tf.image.resize_images(net, size=(256, 256), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        net = tf.layers.conv2d(net,
                               filters=C,
                               kernel_size=(3, 3),
                               padding='same',
                               kernel_initializer=tf.truncated_normal_initializer(0.0, 0.1),
                               activation=tf.nn.elu)
        return net, namescope
