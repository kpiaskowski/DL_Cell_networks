import cv2
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', reshape=False)

epochs = 100
batch_size = 100
n_latent = 20
num_batches = mnist.train.num_examples // batch_size

input_images = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])
label_images = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])

# encoder
e_conv1 = tf.layers.conv2d(input_images, filters=64, kernel_size=4, strides=2, padding='same', activation=tf.nn.leaky_relu)
e_conv2 = tf.layers.conv2d(e_conv1, filters=64, kernel_size=4, strides=2, padding='same', activation=tf.nn.leaky_relu)
e_conv3 = tf.layers.conv2d(e_conv2, filters=64, kernel_size=4, strides=1, padding='same', activation=tf.nn.leaky_relu)
e_flat = tf.contrib.layers.flatten(e_conv3)

# means, stddevs
z_means = tf.layers.dense(e_flat, units=n_latent)
z_stddevs = 0.5 * tf.layers.dense(e_flat, units=n_latent)
epsilon = tf.random_normal(tf.stack([tf.shape(e_flat)[0], n_latent]))
z = z_means + tf.multiply(epsilon, tf.exp(z_stddevs))

# decoder
d_dense1 = tf.layers.dense(z, units=24, activation=tf.nn.leaky_relu)
d_dense2 = tf.layers.dense(d_dense1, units=49, activation=tf.nn.leaky_relu)
d_reshaped = tf.reshape(d_dense2, [-1, 7, 7, 1])

d_conv1 = tf.layers.conv2d(inputs=d_reshaped, filters=64, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
d_upsample1 = tf.image.resize_images(d_conv1, size=(14, 14), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
d_conv2 = tf.layers.conv2d(inputs=d_upsample1, filters=64, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
d_upsample2 = tf.image.resize_images(d_conv2, size=(28, 28), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
d_conv3 = tf.layers.conv2d_transpose(d_upsample2, filters=1, kernel_size=(3, 3), padding='same', activation=tf.nn.sigmoid)

# losses
generative_loss = tf.squeeze(tf.reduce_sum(tf.reduce_sum(tf.squared_difference(d_conv3, label_images), axis=1), axis=1), 1)
latent_loss = -0.5 * tf.reduce_sum(1.0 + 2.0 * z_stddevs - tf.square(z_means) - tf.exp(2.0 * z_stddevs), 1)
loss = tf.reduce_mean(generative_loss + latent_loss)

optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for e in range(epochs):
        for b in range(num_batches):
            batch = mnist.train.next_batch(batch_size)[0]
            _, cost = sess.run([optimizer, loss], feed_dict={input_images: batch, label_images: 1-batch})
            print("Epoch: {}/{}, batch {}/{}, cost: {:.4f}".format(e, epochs, b, num_batches, cost))

        # drawing
        if e % 1 == 0:
            org_imgs = mnist.test.next_batch(batch_size)[0]
            gen_imgs = sess.run(d_conv3, feed_dict={input_images: org_imgs})
            for i in range(10):
                stacked_img = np.hstack([org_imgs[i], gen_imgs[i]])
                cv2.imshow('', stacked_img)
                cv2.waitKey(1000)
