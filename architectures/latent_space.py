import tensorflow as tf


def latent_space(encoder_out, latent_units):
    with tf.variable_scope('latent_space'):
        reduction_conv = tf.layers.conv2d(encoder_out, 256, 1, activation=tf.nn.elu)
        flat = tf.layers.flatten(reduction_conv)

        z_means = tf.layers.dense(flat, units=latent_units)
        z_stddevs = 0.5 * tf.layers.dense(flat, units=latent_units)
        epsilon = tf.random_normal(tf.stack([tf.shape(flat)[0], latent_units]))
        z = z_means + tf.multiply(epsilon, tf.exp(z_stddevs))
        return z, z_means, z_stddevs
