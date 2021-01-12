import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from .parser_tfrec import parser
from .get_pad_and_patch import *
from .generate_label import *
# from augmentations import spatial
import pickle


def pipeline(config, dataset_image_path, dataset_label_path, dataset=None, no_shuffle_and_batching=False):
    """
    Pipeline of tf.data for importing the data
    :param config: type dict,config parameter
    :param dataset_image_path: type str: dataset image path
    :param dataset_label_path: type str: dataset label path
    :return: dataset: return tf.data.dataset: pipeline dataset
    """
    patch_size = config['patch_size']
    dim = len(patch_size)
    # Read max data size of this dataset
    if not config['read_body_identification']:
        max_data_size = [max(config['max_shape']['image'][i], config['max_shape']['label'][i]) for i in range(dim)]
    else:
        max_data_size = [config['max_shape']['image'][i] for i in range(dim)]

    # Choose the channel of the dataset
    input_slice, output_slice = None, None
    if config['input_channel'][dataset] is not None:
        input_slice = config['input_channel'][dataset]
    if config['output_channel'][dataset] is not None:
        output_slice = config['output_channel'][dataset]
        if config['model_add_background_output']:
            # Add background channel at first (0), and all selected channel index +1
            output_slice = [0] + list(map(lambda x: x + 1, output_slice))

    patches_indices = get_fixed_patches_index(config, max_data_size, patch_size,
                                              overlap_rate=config['patch_overlap_rate'],
                                              start=config['patch_start'],
                                              end=config['patch_end'],
                                              max_patch_num=config['max_patch_num'])

    # Reformat data path list: [[path1],[path2], ...] ->[[path1, path2, ...]]
    data_path_image_list = [[t[i] for t in dataset_image_path] for i in range(len(dataset_image_path[0]))]
    data_path_label_list = [[t[i] for t in dataset_label_path] for i in range(len(dataset_label_path[0]))]
    print(data_path_label_list)
    # Create TFRecordDataset list for each image and label path.
    list_image_TFRecordDataset = [tf.data.TFRecordDataset(i) for i in data_path_image_list]
    list_label_TFRecordDataset = [tf.data.TFRecordDataset(i) for i in data_path_label_list]

    # Zip dataset of images and labels.
    zip_data_path_TFRecordDataset = tf.data.Dataset.zip(
        (list_image_TFRecordDataset[0], list_label_TFRecordDataset[0]))

    @tf.function
    def _map(*args):

        """
        Map function of Zip dataset for parsing paths to data
        :param args: args[0] for images TFRecordDataset and args[1] for labels TFRecordDataset
        :return: images_data, labels_data: type list of data tensor
        """
        images_data, images_shape = parser(args[0])
        labels_data, labels_shape = parser(args[1])

        if not config['read_body_identification']:
            # Change orientation
            if config['transpose_permute'] is not None:
                images_data = tf.transpose(images_data, perm=config['transpose_permute'])
                images_shape = tf.transpose(images_shape, perm=config['transpose_permute'])
                labels_data = tf.transpose(labels_data, perm=config['transpose_permute'])
                labels_shape = tf.transpose(labels_shape, perm=config['transpose_permute'])

            # Add background map (the correspond pixel value is zero at all label channels)
            # before the first channel of the label.
            if config['model_add_background_output']:
                # Add background channel at labels_data
                label_sum, label_sum1 = labels_data[..., 0] * 0, labels_data[..., 0] * 0
                label_sum2 = labels_data[..., 0] * 1
                for ch in range(config['channel_label_num'] - 1):
                    label_sum += labels_data[..., ch]
                label_background = label_sum2 - tf.cast(tf.math.greater(label_sum, label_sum1), tf.float32)
                label_background = tf.cast(tf.expand_dims(label_background, axis=-1), tf.float32)
                labels_data = tf.concat([label_background, labels_data], axis=-1)

            # Pad and patch the data
            images_data, labels_data = pad_img_label(config, max_data_size, images_data, images_shape, labels_data,
                                                     labels_shape,
                                                     )

            patchs_imgs, patchs_labels, index_list = get_patches_data(max_data_size, patch_size, images_data,
                                                                      labels_data,
                                                                      patches_indices,
                                                                      slice_channel_img=input_slice,
                                                                      slice_channel_label=output_slice,
                                                                      output_patch_size=config['model_output_size'],
                                                                      random_shift_patch=config['random_shift_patch'])

            # List regularize
            index_list = index_list / (np.array(max_data_size) + 1e-16)
            if config['feed_pos']:
                return (patchs_imgs, index_list), patchs_labels
            else:
                return patchs_imgs, patchs_labels

        else:

            # Pad image size to max shape
            images_data = pad_img_label(config, max_data_size, images_data, images_shape)
            labels_data = tf.pad(tensor=labels_data, paddings=[[0, 0]])

            # Special case for network "body identification"
            if config['transpose_permute'] is not None:
                images_data = tf.transpose(images_data, perm=config['transpose_permute'])
                images_shape = tf.transpose(images_shape, perm=config['transpose_permute'])

            patchs_imgs, _, index_list = get_patches_data(max_data_size, patch_size, images_data,
                                                          data_label=None,
                                                          index_list=patches_indices,
                                                          slice_channel_img=input_slice,
                                                          slice_channel_label=None,
                                                          output_patch_size=config['model_output_size'],
                                                          squeeze_channel=config['squeeze_channel'],
                                                          random_shift_patch=config['random_shift_patch'])
            # Generate labels by patch indices  according to labels_data(position of hip, wrist etc.)
            generate_labels = generate_label(config, labels_data, patches_indices, patch_size)

            if config['model'] == 'model_body_identification_classification':
                # Single output
                generate_labels = generate_labels[0]

            if config['feed_pos']:
                return [patchs_imgs, index_list], generate_labels
            else:
                # Multiple output
                return patchs_imgs, generate_labels  # patchs_imgs, (generate_labels[0], generate_labels[1])

    # Create pipeline and config dataset.
    dataset = zip_data_path_TFRecordDataset.map(map_func=_map, num_parallel_calls=config['num_parallel_calls'])

    if no_shuffle_and_batching:
        dataset = dataset.unbatch()
    else:
        dataset = dataset.unbatch().batch(config['batch']).shuffle(config['shuffle']).prefetch(
            tf.data.experimental.AUTOTUNE)
    """
    while True:
        for elem ,elem2 in dataset:
            print(elem)
            print(elem2)
"""

    return dataset
