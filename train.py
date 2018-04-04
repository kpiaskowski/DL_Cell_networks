import argparse
import os

import tensorflow as tf
from architectures.conv_decoder import conv_decoder
from architectures.pretrained_encoder import pretrained_encoder as encoder

from constants import img_w, img_h
from dataprovider_train import DataProvider
from utils import create_training_dirs, draw_tensorboard_predictions

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains models')
    parser.add_argument('--model_name', help='name of model', dest='model_name')
    parser.add_argument('--epochs', help='number of epochs', dest='epochs')
    parser.add_argument('--l_rate', help='learning rate', dest='l_rate')
    parser.add_argument('--thresh', help='threshold, default value = 0.3', dest='thresh')
    parser.add_argument('--batch_size', help='size of single batch', dest='batch_size')
    parser.add_argument('--saver_checkpoint', help='save every n iterations', dest='saver_checkpoint')
    parser.add_argument('--load_checkpoint', help='if you continue training, which checkpoint to load', dest='load_checkpoint')

    parser.add_argument('--t_img_path', help='path to folder with train images', dest='t_img_path')
    parser.add_argument('--v_img_path', help='path to folder with validation images', dest='v_img_path')
    parser.add_argument('--t_label_path', help='path to folder with train labels', dest='t_label_path')
    parser.add_argument('--v_label_path', help='path to folder with validation labels', dest='v_label_path')

    args = parser.parse_args()

    # experiment params
    model_name = args.model_name
    epochs = float(args.epochs)
    l_rate = float(args.l_rate)
    thresh = float(args.thresh)
    batch_size = int(args.batch_size)
    logging_checkpoint = 20
    saver_checkpoint = int(args.saver_checkpoint)
    load_checkpoint = int(args.load_checkpoint) if args.load_checkpoint else saver_checkpoint

    # paths
    t_img_path = args.t_img_path
    v_img_path = args.v_img_path
    t_label_path = args.t_label_path
    v_label_path = args.v_label_path

    if any((i is None for i in (model_name, epochs, l_rate, thresh, batch_size, saver_checkpoint,
                                t_img_path, t_label_path, v_img_path, v_label_path))):
        raise Exception('Provide all parameters (images_path, annotations_path, labels_path, dst_h, dst_w)!')

    # data
    t_num_batches = len(os.listdir(t_label_path)) // batch_size
    v_num_batches = len(os.listdir(v_label_path)) // batch_size

    dataset = DataProvider(batch_size)
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
                print("Training, epoch: {} of {}, batch: {} of {}, cost: {}".format(i_t // t_num_batches + 1, epochs, i_t % t_num_batches + 1, t_num_batches,
                                                                                    cost))
                i_t += 1

            _, cost, summary = sess.run([train_op, loss, merged], feed_dict={handle: training_handle})
            print("Training, epoch: {} of {}, batch: {} of {}, cost: {}".format(i_t // t_num_batches + 1, epochs, i_t % t_num_batches + 1, t_num_batches, cost))
            train_writer.add_summary(summary, i_t)
            train_writer.flush()
            i_t += 1

            cost, summary = sess.run([loss, merged], feed_dict={handle: validation_handle})
            print(
                "Validation, epoch: {} of {}, batch: {} of {}, cost: {}".format(i_t // t_num_batches + 1, epochs, i_t % t_num_batches + 1, t_num_batches, cost))
            val_writer.add_summary(summary, i_t)
            val_writer.flush()
            i_v += 1

            if i_t % saver_checkpoint < logging_checkpoint:
                saver.save(sess, 'saved_models/' + model_name + '/model.ckpt', global_step=i_t)
