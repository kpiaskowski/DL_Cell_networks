import os
import xml.etree.ElementTree as ET

import numpy as np
import tensorflow as tf


class DataProvider:
    """Generic dataprovider for object detection tasks"""

    def __init__(self, img_h, img_w, h, w, batch_size):
        """
        Initializes DataProvider
        :param img_h: the desired height of input images (they will be resized to this)
        :param img_w: the desired width of input images (they will be resized to this)
        :param h: desired height of the occupancy tensor
        :param w: desired width of the occupancy tensor
        :param batch_size: size of batch
        """
        self.img_h = img_h
        self.img_w = img_w
        self.h = h
        self.w = w
        self.batch_size = batch_size

        # these variables are dataset dependent and will be created in child classes
        self.classes = None
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def _create_occupancy_tensor(self, object_list, img_h, img_w, h, w, c):
        """
        Creates an occupancy tensor, based on object list
        :param object_list: : a list of tuples in form: (class_name, (xmin, ymin, xmax, ymax))
        :param img_h: height of image
        :param img_w: width of image
        :param h: desired height of occupance tensor
        :param w: desired width of occupance tensor
        :param c: number of classes in occupance tensor
        :return: occupancy tensor
        """
        tensor = np.zeros([h, w, c], np.uint8)

        # fill tensor
        for obj_info in object_list:
            # index of class and unnormalized dimensions of object
            class_idx = self.classes.index(obj_info[0])
            xmin, ymin, xmax, ymax = obj_info[1]

            # normalize dimensions with regard to image size
            xmin = int(xmin / img_w * w)
            xmax = int(xmax / img_w * w)
            ymin = int(ymin / img_h * h)
            ymax = int(ymax / img_h * h)

            # fill the tensor with
            tensor[ymin: ymax, xmin: xmax, class_idx] = 1

        return tensor

    def _annotation_to_tensor(self, annotation, h, w, c):
        """
        Parses annotation information and converts it into an occupancy tensor of shape [h, w, num_classes]
        :param annotation: annotation information, dependent on dataset
        :param h: desired tensor height
        :param w: desired tensor width
        :param c: number of classes
        :return a tensor of shape [h,c,c], filled with ones, where objects are present and zeros otherwise
        """
        raise NotImplementedError

    def _pair_images_with_annotations(self, **kwargs):
        """
        Returns two lists: list of image names and list of corresponding annotations; for training and validation.
        :param kwargs: dependent on dataset, could be a list of files, paths (or anything else) which helps to relate imgs to their annotations
        """
        raise NotImplementedError

    def _tf_decode_images(self, image):
        """
        Loads image in Tensorflow DatasetAPI
        :param image: image data (for example path), dependent on dataset
        :return: decoded image
        """
        image_string = tf.read_file(image)
        image_decoded = tf.image.decode_jpeg(image_string)
        image_resized = tf.image.resize_images(image_decoded, [self.img_h, self.img_w])
        return image_resized

    def _tf_decode_annotation(self, annotation):
        """
        Read and decode annotation, depending on dataset
        :param annotation: annotation data (for example path), dependent on dataset
        :return: decoded annotation
        """
        raise NotImplementedError

    def _tf_define_dataset(self, image_names, annotation_data=None):
        """
        Creates TF dataset using DatasetAPI
        :param image_data: list of data snippets related to images (for example paths), dependent on dataset and tf_decode[...] functions
        :param annotation_data: list of data snippets related to images (for example paths), dependent on dataset and tf_decode[...] functions
        :param num_epochs: number of epochs: if none, then run infinitely times
        :return: TF dataset
        """
        # for train and validation datasets
        if annotation_data:
            tf_image_names = tf.constant(image_names)
            tf_annotation_data = tf.constant(annotation_data)
            dataset = tf.data.Dataset.from_tensor_slices((tf_image_names, tf_annotation_data))
            dataset = dataset.shuffle(buffer_size=5000)
            dataset = dataset.map(lambda i, a: (self._tf_decode_images(i), a), num_parallel_calls=8)
            dataset = dataset.map(lambda i, a: (i, tf.py_func(self._tf_decode_annotation, [a], tf.uint8, stateful=False)), num_parallel_calls=8)
            dataset = dataset.prefetch(self.batch_size)
            dataset = dataset.batch(self.batch_size)
            dataset = dataset.repeat()
            return dataset

        # for test set, we add image names for further performance evaluation on online websites
        else:
            tf_image_names = tf.constant(image_names)
            dataset = tf.data.Dataset.from_tensor_slices((tf_image_names, tf_image_names)) # the second 'names' arg will serve as a pointer to the file during evaluation
            # dataset = dataset.map(self._tf_decode_images, num_parallel_calls=8)
            dataset = dataset.map(lambda i, a: (self._tf_decode_images(i), a), num_parallel_calls=8)
            dataset = dataset.prefetch(self.batch_size)
            dataset = dataset.batch(self.batch_size)
            dataset = dataset.repeat(1)
            return dataset

    def train_val_dataset(self):
        raise NotImplementedError

    def test_dataset(self):
        """
        Create TF Dataset for for testing, using one shot iterator (no initialization needed)
        :return: handle to test dataset (only images)
        """
        raise NotImplementedError


class PascalProvider(DataProvider):
    """Dataprovider for PASCAL VOC dataset. Combines data from 2007 and 2012"""

    def __init__(self, training_dirs, validation_dir, test_dir, img_h, img_w, h, w, batch_size):
        """
        Initializes PascalProvider. You should have 4 directories with data: VOC2007_test (for validation), VOC2007_train, VOC2012_train (for training) and todo add test.
        Note that after downloading data from PascalVOC site, these dirs are named differently. You need to rename them by yourself. See below to find out how
        to map Pascal data into train/validation/test data. Also, do not change anything within the folders themselves (the structure: Annotations, ImageSets,
        JPEGImages ... ) should be preserved). For final testing, we use data from PascalVOC2012, specified as test data, and evaluate it online
        :param training_dirs: tuple containing paths to root training directories, for 2007 and 2012. We use trainval data, which means that we use all data from 2007 and 2012 for training.
        :param validation_dir: path to root test directory. For validation (and final testing) we use data from PascalVOC 2007 test set.
        :param test_dir: we use test data from Pascal2012
        """
        super().__init__(img_h, img_w, h, w, batch_size)

        # dataset specific info
        self.classes = ["person", "bird", "cat", "cow", "dog", "horse", "sheep", "aeroplane", "bicycle", "boat", "bus", "car", "motorbike", "train",
                        "bottle", "chair", "diningtable", 'pottedplant', "sofa", "tvmonitor"]
        self.num_classes = len(self.classes)

        # data in form: list(image paths), list(annotation paths), except for test data, where only list(image paths) is provided
        self.train_data, self.val_data = self._pair_images_with_annotations(training_dirs=training_dirs, validation_dir=validation_dir)
        self.test_data = self.val_data[0]  # todo change when website with test data will finally work

    def _pair_images_with_annotations(self, **kwargs):
        """
        Finds all pairs (image, annotation as XML file) for training and validation.
        :param kwargs: a dictionary with keys: 'training_dirs', 'validation_dir', describing paths to the respective datasets. Train is actually a tuple of paths, containing data from Pascal2007 and Pascal2012
        :return 2 list of names for train and validation set.
        """

        # create a subfunction, used only here to ease the process of reading names
        def read_names(root_dir):
            """
            Read data from root folder
            :param root_dir: A folder containing PascalVOC subfolders: Annotations, ImageSets, JPEGImages and so on
            :return: two lists: a list of image names and corresponding list of annotation names
            """
            # directories, created only to avoid excessive usage of os.path.join
            images_dir = os.path.join(root_dir, 'JPEGImages')
            annotations_dir = os.path.join(root_dir, 'Annotations')

            # read names of images and XML annotations
            image_names = sorted([os.path.join(images_dir, name) for name in os.listdir(images_dir)])
            annotation_names = sorted([os.path.join(annotations_dir, name) for name in os.listdir(annotations_dir)])

            return image_names, annotation_names

        # training
        training = [[], []]
        for root_dir in kwargs['training_dirs']:
            image_names, annotation_names = read_names(root_dir)
            training[0].extend(image_names)
            training[1].extend(annotation_names)

        # validation
        validation = [[], []]
        image_names, annotation_names = read_names(kwargs['validation_dir'])
        validation[0].extend(image_names)
        validation[1].extend(annotation_names)

        return training, validation

    def _tf_decode_annotation(self, annotation):
        """
        Read and decode annotation, then transform it into the occupancy tensor
        :param annotation: a path to an XML annotation
        :return: annotation transformed into occupancy tensor
        """
        occupancy_tensor = self._annotation_to_tensor(annotation, self.h, self.w, self.num_classes)
        return occupancy_tensor

    def _annotation_to_tensor(self, xml_file, h, w, c):
        """
        Parses annotation information (XML file in PASCAL) and converts it into an occupancy tensor of shape [h, w, num_classes]
        :param h: desired tensor height
        :param w: desired tensor width
        :param c: number of classes
        :return a tensor of shape [h,c,c], filled with ones, where objects are present and zeros otherwise
        """
        # create the tree from file
        tree = ET.parse(xml_file)

        # find dimensions of associated image, for position normalization
        size = tree.find('size')
        img_w = int(size.find('width').text)
        img_h = int(size.find('height').text)

        # find all objects
        object_list = []  # list of objects, described by a nested tuple: (class_name, (xmin, ymin, xmax, ymax))
        objects = tree.findall('object')
        for object in objects:
            cls = object.find('name').text
            bbox = object.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            xmax = int(bbox.find('xmax').text)
            ymin = int(bbox.find('ymin').text)
            ymax = int(bbox.find('ymax').text)

            object_list.append((cls, (xmin, ymin, xmax, ymax)))

        # create the ocupancy tensor
        return self._create_occupancy_tensor(object_list, img_h, img_w, h, w, c)

    def train_val_dataset(self):
        """
        Create TF Dataset for training and validation. Remember to initialize respective hooks manually.
        :return: handles for both datasets and data hooks (images and occupancy tensors)
        """
        # feedable iterators for train and validation
        train_dataset = self._tf_define_dataset(self.train_data[0], self.train_data[1])
        val_dataset = self._tf_define_dataset(self.val_data[0], self.val_data[1])

        # iterators for dataset
        handle = tf.placeholder(tf.string, shape=[])
        iter = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)
        images, occupancy_tensors = iter.get_next()
        train_iter = train_dataset.make_one_shot_iterator()
        val_iter = val_dataset.make_one_shot_iterator()

        return handle, train_iter, val_iter, images, occupancy_tensors

    def test_dataset(self):
        """
        Create TF Dataset for for testing, using one shot iterator
        :return: handle to test dataset (only images) and its iterator
        """
        test_dataset = self._tf_define_dataset(self.test_data, None)
        iterator = test_dataset.make_initializable_iterator()
        images, filenames = iterator.get_next()
        return images, filenames, iterator


class COCOProvider(DataProvider):
    """Dataprovider for Ms COCO dataset (from 2017)"""

    def __init__(self, training_dir, validation_dir, test_dir, annotations_dir, img_h, img_w, h, w, batch_size):
        """
        Initializes COCO dataset.
        :param training_dir: path to COCO 'train2017' dir
        :param validation_dir: path to COCO 'val2017' dir
        :param test_dir: path to COCO 'test2017' dir
        :param annotations_dir: path to COCO 'annotations' dir
        :param img_h: desired image height (images will be resized to this height)
        :param img_w: desired image width (images will be resized to this width)
        :param h: desired occupancy tensor height
        :param w: desired occupancy tensor width
        :param batch_size: the size of batch
        """
        super().__init__(img_h, img_w, h, w, batch_size)


provider = COCOProvider(training_dir='../data/MsCOCO/train2007',
                        validation_dir='../data/MsCOCO/val2007',
                        test_dir='../data/MsCOCO/test2007',
                        annotations_dir='../data/MsCOCO/annotations',
                        img_h=448, img_w=448, h=28, w=28,
                        batch_size=10)


# an example of how to use and how the data is shaped
if __name__ == '__main__':
    # pascal data
    pascal_provider = PascalProvider(training_dirs=('../data/PascalVOC/VOC2007_train', '../data/PascalVOC/VOC2012_train'),
                                     validation_dir='../data/PascalVOC/VOC2007_test',
                                     test_dir=None,  # todo
                                     img_h=448, img_w=448, h=28, w=28,
                                     batch_size=10)

    handle, train_iter, val_iter, images, occupancy_tensors = pascal_provider.train_val_dataset()
    test_images, test_filenames, test_iter = pascal_provider.test_dataset()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # PASCAL data - train/val
        train_handle, val_handle = sess.run([train_iter.string_handle(), val_iter.string_handle()])

        imgs, occ_tensor = sess.run([images, occupancy_tensors], feed_dict={handle: train_handle})
        print('PASCAL (train and val): img shape: {}, occup_tensor shape: {}'.format(imgs.shape, occ_tensor.shape))

        # PASCAL data - test
        sess.run(test_iter.initializer)
        imgs, names = sess.run([test_images, test_filenames])
        print('PASCAL (test): img shape: {}, names: {}'.format(imgs.shape, names.shape))
