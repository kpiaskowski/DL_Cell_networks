import os

import tensorflow as tf
from architectures.conv_decoder import conv_decoder
from architectures.pretrained_encoder import pretrained_encoder as encoder

from constants import img_w, img_h
from dataprovider_train import DataProvider
from utils import create_training_dirs, draw_tensorboard_predictions

# experiment params
epochs = 100
l_rate = 0.00001
thresh = 0.3
batch_size = 15
model_name = 'conv_model'
logging_checkpoint = 20
saver_checkpoint = 1500
load_checkpoint = saver_checkpoint

# paths
t_img_path = 'data/train2017'
v_img_path = 'data/val2017'
t_label_path = 'data/train_labels_S14'
v_label_path = 'data/val_labels_S14'

# data
t_num_batches = len(os.listdir(t_label_path)) // batch_size
v_num_batches = len(os.listdir(v_label_path)) // batch_size

dataset = DataProvider('conv_decoder', batch_size)
handle, training_iterator, validation_iterator, images, labels = dataset.get_data(t_img_path, v_img_path, t_label_path, v_label_path)

# model
encoder = encoder(images)
logits = conv_decoder(encoder)
output = tf.nn.sigmoid(logits)

# losses
loss = tf.losses.sigmoid_cross_entropy(labels, logits=logits)
train_op = tf.train.AdamOptimizer(l_rate).minimize(loss)

# saving and logging
create_training_dirs('saved_models', 'saved_summaries', model_name)
saver = tf.train.Saver(max_to_keep=10, keep_checkpoint_every_n_hours=3)
pretrained_loader = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='yolo'))

# summaries
with tf.name_scope('summaries'):
    boxed_predictions = tf.expand_dims(tf.py_func(draw_tensorboard_predictions, [images, tf.cast(tf.to_int32(output > thresh), tf.float32)], tf.float32), 0)
    boxed_labels = tf.expand_dims(tf.py_func(draw_tensorboard_predictions, [images, tf.cast(tf.to_int32(labels > thresh), tf.float32)], tf.float32), 0)

    output_masks = tf.gather(output, [0])
    output_masks = tf.reduce_max(output_masks, -1, keep_dims=True)
    output_masks = tf.image.resize_images(output_masks, [img_h, img_w])
    output_masks = tf.image.grayscale_to_rgb(output_masks)
    output_masks = tf.cast(tf.to_int32(output_masks > thresh), tf.float32)

    concat_images = tf.concat([boxed_labels, boxed_predictions, output_masks], axis=2)

    tf.summary.image('outputs', concat_images, max_outputs=5)
    tf.summary.scalar('loss', loss)
merged = tf.summary.merge_all()

with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(os.path.join('saved_summaries', model_name, 'train'), sess.graph, flush_secs=30)
    val_writer = tf.summary.FileWriter(os.path.join('saved_summaries', model_name, 'validation'), flush_secs=30)
    training_handle, validation_handle = sess.run([training_iterator.string_handle(), validation_iterator.string_handle()])

    sess.run(tf.global_variables_initializer())

    if len(os.listdir('saved_models/' + model_name)) > 0:
        saver.restore(sess, os.path.join('saved_models', model_name, 'model.ckpt-' + str(load_checkpoint)))
        print(model_name + ' loaded')
    else:
        pretrained_loader.restore(sess, 'pretrained_imagenet/pretrained_imagenet.ckpt')  # load encoder pretrained on imagenet
        print('Training %s from scratch' % model_name)

    i_t = 0
    i_v = 0
    while True:
        for _ in range(logging_checkpoint - 1):
            _, cost = sess.run([train_op, loss], feed_dict={handle: training_handle})
            print("Training, epoch: {} of {}, batch: {} of {}, cost: {}".format(i_t // t_num_batches + 1, epochs, i_t % t_num_batches + 1, t_num_batches, cost))
            i_t += 1

        _, cost, summary = sess.run([train_op, loss, merged], feed_dict={handle: training_handle})
        print("Training, epoch: {} of {}, batch: {} of {}, cost: {}".format(i_t // t_num_batches + 1, epochs, i_t % t_num_batches + 1, t_num_batches, cost))
        train_writer.add_summary(summary, i_t)
        train_writer.flush()
        i_t += 1

        cost, summary = sess.run([loss, merged], feed_dict={handle: validation_handle})
        print("Validation, epoch: {} of {}, batch: {} of {}, cost: {}".format(i_t // t_num_batches + 1, epochs, i_t % t_num_batches + 1, t_num_batches, cost))
        val_writer.add_summary(summary, i_t)
        val_writer.flush()
        i_v += 1

        if i_t % saver_checkpoint < logging_checkpoint:
            saver.save(sess, 'saved_models/' + model_name + '/model.ckpt', global_step=i_t)
