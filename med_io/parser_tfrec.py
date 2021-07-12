import tensorflow as tf


def parser(example_serialized):
    """
    Parse paths to data
    :param example_serialized: type scalar string Tensor, A single serialized Example.
    :return: images: type Tensor: Image data
    """
    # all the images and labels tfrecords files must have same features key 'image' and 'image_shape'!
    # features['image'] must be tf.float and features['image_shape'] must be tf.int32!
    features_map = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'image_shape': tf.io.FixedLenFeature([], tf.string)
    }
    features = tf.io.parse_single_example(example_serialized, features_map)
    images = tf.io.decode_raw(features['image'], tf.float32)

    images_shape = tf.io.decode_raw(features['image_shape'], tf.int32)
    images = tf.reshape(images, images_shape)
    return images, images_shape

def parser_melanom(example_serialized):
    """
    Parse paths to data from dataset MELANOM
    :param example_serialized: type scalar string Tensor, A single serialized Example.
    :return: images: type Tensor: Image data
    """
    # all the images and labels tfrecords files must have same features key 'image' and 'image_shape'!
    # features['image'] must be tf.float and features['image_shape'] must be tf.int32!
    features_map = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'image_shape': tf.io.FixedLenFeature([], tf.string),
        'validation_for_cancer': tf.io.FixedLenFeature([], tf.int64)
    }
    features = tf.io.parse_single_example(example_serialized, features_map)
    images = tf.io.decode_raw(features['image'], tf.float32)

    images_shape = tf.io.decode_raw(features['image_shape'], tf.int32)
    validation_for_cancer = features['validation_for_cancer']


    images = tf.reshape(images, images_shape)
    return images, images_shape, validation_for_cancer