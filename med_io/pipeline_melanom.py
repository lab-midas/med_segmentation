import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from .parser_tfrec import parser
from .get_pad_and_patch import *
from .generate_label import *
from .augmentation import *
import pickle
from tensorflow.keras import layers


def pipeline_melanom(config, dataset_image_path, dataset_label_path, dataset=None, augment=False):
    """
    Pipeline of tf.data for importing the data
    :param config: type dict,config parameter
    :param dataset_image_path: type str: dataset image path
    :param dataset_label_path: type str: dataset label path
    :return: dataset: return tf.data.dataset: pipeline dataset
    """
    patch_size = config['patch_size']
    print("patch size: ", patch_size)
    dim = len(patch_size)
    print("dim: ", dim)
    # Read max data size of this dataset
    if not config['read_body_identification']:
        max_data_size = [max(config['max_shape']['image'][i], config['max_shape']['label'][i]) for i in range(dim)]
    else:
        max_data_size = [config['max_shape']['image'][i] for i in range(dim)]

    print("printing max data size")
    print(max_data_size)

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
                                              end=config['patch_end'])

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

    #print("zip: ", next(zip_data_path_TFRecordDataset))

    @tf.function
    def _map(*args):

        """
        Map function of Zip dataset for parsing paths to data
        :param args: args[0] for images TFRecordDataset, args[1] for labels TFRecordDataset
        :return: images_data, labels_data: type list of data tensor
        """
        images_data, images_shape = parser(args[0])
        print("images_shape: ", images_shape)
        labels_data, labels_shape = parser(args[1])
        print("labels_shape: ", labels_shape)

        # Pad and patch the data
        images_data, labels_data = pad_img_label(config, max_data_size, images_data, images_shape, labels_data,
                                                 labels_shape)

        print("image type: ", type(images_data))
        print("image type len : ", len(images_data))
        print("image first: ", images_data[0].shape)
        print(images_data)

        patchs_imgs, patchs_labels, index_list = get_patches_data(max_data_size, patch_size, images_data,
                                                                  labels_data,
                                                                  patches_indices, slice_channel_img=input_slice,
                                                                  slice_channel_label=output_slice,
                                                                  output_patch_size=config['model_output_size'])
        print("patch type: ", type(patchs_imgs))
        print("patch shape: ", patchs_imgs[0].shape)
        print(patchs_imgs)

        ## List regularize
        index_list = index_list / (np.array(max_data_size) + 1e-16)
        if config['feed_pos']:
            return (patchs_imgs, index_list), patchs_labels
        else:
            return patchs_imgs, patchs_labels

    # Create pipeline and config dataset.
    # dataset = zip_data_path_TFRecordDataset.map(map_func=_map, num_parallel_calls=config['num_parallel_calls'])
    dataset = zip_data_path_TFRecordDataset.map(map_func=_map, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    for elem in dataset.take(2):
        print("image size: ", elem[0].shape)
        print("label size: ", elem[1].shape)

    @tf.function
    def data_augmentation(images_data, labels_data):
        ####----------------------------------
        ##augmentation
        # this does not correspond to pytorch architecture
        ## augmentation is performed after patching
        ## augmentations used are contrast and brightness change: gamma, brightness, contrast
        ## further details implemented in PyTorch archictecture

        ###------------------------------------

        #images_data = args[0]
        print("images data: ", images_data.numpy().shape)
        #labels_data = args[1]
        print("label data: ",labels_data.numpy().shape)

        ##assert len(images_data.shape) == 4

        transformation_list = config['augmentation']

        for transform in transformation_list:
            print("transformation: ", transform)

        for transformation in transformation_list:
            if transformation == 'brightness':
                images_data = brightness_transform(images_data, mu=0.0, sigma=0.3)

            if transformation == 'gamma':
                images_data = gamma_contrast(images_data, gamma_range=(0.7, 1.3))

            if transformation == 'contrast':
                images_data = contrast_augmentation_transform(images_data, contrast_range=(0.3, 1.7))

        return images_data, labels_data

    if augment:
        dataset = dataset.map(map_func=data_augmentation, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    print("dataset type: ", type(dataset))

    dataset = dataset.unbatch().shuffle(config['shuffle']).batch(config['batch']).prefetch(16)
    # tf.data.experimental.AUTOTUNE)

    print("dataset type: ", type(dataset))

    # while True:
    # for elem ,elem2 in dataset:
    # print("elem: ", elem)
    # print("elem2: ", elem2)

    return dataset
