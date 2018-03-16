import os

import tensorflow as tf
from architectures.conv_decoder import conv_decoder
from architectures.pretrained_encoder import pretrained_encoder as encoder

from constants import batch_size
from dataprovider import DataProvider
from utils import create_training_dirs, debug_and_save_imgs

# experiment params
epochs = 30
l_rate = 0.00001
thresh = 0.3
model_name = 'test_refactor'
save_img_checkpoint = 200
decoder = conv_decoder
mode = conv_decoder.__name__

t_img_path = 'data/train2017'
v_img_path = 'data/val2017'

t_label_path = 'data/train_labels_S14'
v_label_path = 'data/val_labels_S14'

# data
t_num_batches = len(os.listdir(t_label_path)) // batch_size
v_num_batches = len(os.listdir(v_label_path)) // batch_size
t_num_batches -= t_num_batches // save_img_checkpoint + 1
v_num_batches -= v_num_batches // save_img_checkpoint + 1

dataset = DataProvider(mode)
training_dataset_init, validation_dataset_init, images, labels = dataset.get_data(t_img_path, v_img_path, t_label_path, v_label_path)

encoder = encoder(images)
logits, decoder_namescope = decoder(encoder)

output = tf.nn.sigmoid(logits)

loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
loss = tf.reduce_mean(loss)

train_op = tf.train.AdamOptimizer(l_rate).minimize(loss)

# saving and logging
create_training_dirs('saved_models', 'saved_summaries', 'generated_images', model_name)
saver_decoder = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=decoder_namescope))
saver_encoder = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='yolo'))

with tf.name_scope('summaries'):
    tf.summary.scalar('loss', loss)
merged = tf.summary.merge_all()

with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(os.path.join('saved_summaries', model_name, 'train'), sess.graph)
    val_writer = tf.summary.FileWriter(os.path.join('saved_summaries', model_name, 'validation'))
    sess.run(tf.global_variables_initializer())

    if len(os.listdir(os.path.join('saved_models', model_name))) > 0:
        saver_decoder.restore(sess, os.path.join('saved_models', model_name, 'd_model.ckpt'))
        saver_encoder.restore(sess, os.path.join('saved_models', model_name, 'e_model.ckpt'))
        print(model_name + ' loaded')
    else:
        saver_encoder.restore(sess, 'pretrained_imagenet/pretrained_imagenet.ckpt')  # load encoder pretrained on imagenet
        print('Training %s from scratch' % model_name)

    for epoch in range(epochs):
        # training
        sess.run(training_dataset_init)
        for k in range(t_num_batches):
            _, cost, summary = sess.run([train_op, loss, merged])
            print("\rTraining, epoch: {} of {}, batch: {} of {}, cost: {}".format(epoch + 1, epochs, k + 1, t_num_batches, cost), end='', flush=True)
            train_writer.add_summary(summary, epoch * t_num_batches + k)

            if k % save_img_checkpoint == 0:
                for i in range(1):
                    masks, imgs, gt_masks = sess.run([output, images, labels])
                    mask = masks[i]
                    img = imgs[i]
                    gt_mask = gt_masks[i]
                    filepath = os.path.join('generated_images', model_name, '_e_' + str(epoch + 1) + 'k_' + str(k + 1) + '_i_' + str(i) + '_train.jpg')
                    debug_and_save_imgs(img, mask, gt_mask, thresh, filepath)

        saver_decoder.save(sess, os.path.join('saved_models', model_name, 'd_model.ckpt'))
        saver_encoder.save(sess, os.path.join('saved_models', model_name, 'e_model.ckpt'))
        print()

        # validation
        sess.run(validation_dataset_init)
        for k in range(v_num_batches):
            cost, summary = sess.run([loss, merged])
            print("\rTesting, epoch: {} of {}, batch: {} of {}, cost: {}".format(epoch + 1, epochs, k + 1, v_num_batches, cost), end='', flush=True)
            val_writer.add_summary(summary, epoch * v_num_batches + k)

            if k % save_img_checkpoint == 0:
                for i in range(1):
                    masks, imgs, gt_masks = sess.run([output, images, labels])
                    mask = masks[i]
                    img = imgs[i]
                    gt_mask = gt_masks[i]
                    filepath = os.path.join('generated_images', model_name, '_e_' + str(epoch + 1) + 'k_' + str(k + 1) + '_i_' + str(i) + '_val.jpg')
                    debug_and_save_imgs(img, mask, gt_mask, thresh, filepath)
        print()
