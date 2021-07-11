import tensorflow as tf
import numpy as np
import os

def write_tfrecord(data, path):
    """
    writing data to a single tf_record to the path
    saving images and images shape only.
    :param data: type ndarray:  Image data
    :param path: type str: Path to tfrecord file
    :return:
  """


    with tf.io.TFRecordWriter(path) as writer:
            example = tf.train.Example(features=tf.train.Features(feature={
                "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[data.tobytes()])),
                'image_shape': tf.train.Feature(bytes_list=tf.train.BytesList(value=[np.int32(np.array(data.shape)).tobytes()])),
            }))
            writer.write(example.SerializeToString())

    writer.close()

def write_tfrecord_melanoma(data, validation_for_cancer, path):
    """
    writing data to a single tf_record to the path
    saving images and images shape only.

    :param data: type ndarray:  Image data
    :param validation_for_cancer: variable that indicates if patient has cancer or not
    :param path: type str: Path to tfrecord file
    :return:
  """


    with tf.io.TFRecordWriter(path) as writer:
            example = tf.train.Example(features=tf.train.Features(feature={
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data.tobytes()])),
                'image_shape': tf.train.Feature(bytes_list=tf.train.BytesList(value=[np.int32(np.array(data.shape)).tobytes()])),
                'validation_for_cancer': tf.train.Feature(int64_list=tf.train.Int64List(value=[validation_for_cancer]))
            }))
            writer.write(example.SerializeToString())

    writer.close()