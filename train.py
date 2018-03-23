import os

import tensorflow as tf
from architectures.autoencoder import ae_decoder
from architectures.conv_decoder import conv_decoder
from architectures.pretrained_encoder import pretrained_encoder as encoder

from constants import batch_size, img_w, img_h
from dataprovider import DataProvider
from utils import create_training_dirs, debug_and_save_imgs

# experiment params
epochs = 50
l_rate = 0.00001
thresh = 0.3
model_name = 'model5'
save_img_checkpoint = 20
architecture = 'conv'
saver_checkpoint = 30

t_img_path = 'data/train2017'
v_img_path = 'data/val2017'

t_label_path = 'data/train_labels_S14'
v_label_path = 'data/val_labels_S14'

if architecture == 'conv':
    decoder = conv_decoder
if architecture == 'autoencoder':
    decoder = ae_decoder

mode = decoder.__name__

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
saver = tf.train.Saver(max_to_keep=10, keep_checkpoint_every_n_hours=2)
pretrained_loader = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='yolo'))

with tf.name_scope('summaries'):
    output_masks = tf.reduce_max(output, -1, keep_dims=True)
    output_masks = tf.image.resize_images(output_masks, [img_h, img_w])
    output_masks = tf.image.grayscale_to_rgb(output_masks)
    concat_images = tf.concat([images, output_masks], axis=2)
    tf.summary.image('outputs', concat_images, max_outputs=5)
    tf.summary.scalar('loss', loss)
merged = tf.summary.merge_all()

with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(os.path.join('saved_summaries', model_name, 'train'), sess.graph, flush_secs=30)
    val_writer = tf.summary.FileWriter(os.path.join('saved_summaries', model_name, 'validation'), flush_secs=30)
    sess.run(tf.global_variables_initializer())

    if len(os.listdir('saved_models/' + model_name)) > 0:
        saver.restore(sess, os.path.join('saved_models', model_name, 'model.ckpt'))
        print(model_name + ' loaded')
    else:
        pretrained_loader.restore(sess, 'pretrained_imagenet/pretrained_imagenet.ckpt')  # load encoder pretrained on imagenet
        print('Training %s from scratch' % model_name)

    for epoch in range(epochs):
        # training
        sess.run(training_dataset_init)
        i = 0
        while True:
            try:
                _, cost, summary = sess.run([train_op, loss, merged])
                print("Training, epoch: {} of {}, batch: {} of {}, cost: {}".format(epoch + 1, epochs, i + 1, t_num_batches, cost))
                train_writer.add_summary(summary, epoch * t_num_batches + i)
                i += 1

                if i % save_img_checkpoint == 0:
                    for k in range(1):
                        masks, imgs, gt_masks = sess.run([output, images, labels])
                        mask = masks[k]
                        img = imgs[k]
                        gt_mask = gt_masks[k]
                        filepath = os.path.join('generated_images', model_name, 'train_e_' + str(epoch + 1) + 'k_' + str(i + 1) + '_i_' + str(i) + '.jpg')
                        debug_and_save_imgs(img, mask, gt_mask, thresh, filepath)
                if i % saver_checkpoint == 0:
                    saver.save(sess, 'saved_models/' + model_name + '/model.ckpt', global_step=epoch * t_num_batches + i)
            except tf.errors.OutOfRangeError:
                break

        # validation
        sess.run(validation_dataset_init)
        i = 0
        while True:
            try:
                cost, summary = sess.run([loss, merged])
                print("Validation, epoch: {} of {}, batch: {} of {}, cost: {}".format(epoch + 1, epochs, i + 1, v_num_batches, cost))
                val_writer.add_summary(summary, epoch * v_num_batches + i)
                i += 1
                if i % save_img_checkpoint == 0:
                    for i in range(1):
                        masks, imgs, gt_masks = sess.run([output, images, labels])
                        mask = masks[i]
                        img = imgs[i]
                        gt_mask = gt_masks[i]
                        filepath = os.path.join('generated_images', model_name, 'val_e_' + str(epoch + 1) + 'k_' + str(i + 1) + '_i_' + str(i) + '.jpg')
                        debug_and_save_imgs(img, mask, gt_mask, thresh, filepath)
            except tf.errors.OutOfRangeError:
                break
