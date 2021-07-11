import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from .parser_tfrec import parser, parser_melanom
from .get_pad_and_patch import *
from .generate_label import *
# from augmentations import spatial
import pickle
from .augmentation import *

def pipeline(config, dataset_image_path, dataset_label_path, dataset=None, no_shuffle_and_batching=False):
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
                                                     labels_shape)

            patchs_imgs, patchs_labels, index_list = get_patches_data(max_data_size, patch_size, images_data,
                                                                      labels_data,
                                                                      patches_indices,
                                                                      slice_channel_img=input_slice,
                                                                      slice_channel_label=output_slice,
                                                                      output_patch_size=config['model_output_size'],
                                                                      random_shift_patch=config['random_shift_patch'],
                                                                      squeeze_channel=config['squeeze_channel'])


            # List regularize
            #index_list = index_list / (np.array(max_data_size) + 1e-16)
            if config['active_learning']:
                return patchs_imgs, patchs_labels, index_list
            elif config['feed_pos']:
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
                return patchs_imgs, generate_labels

    # Create pipeline and config dataset.
    dataset = zip_data_path_TFRecordDataset.map(map_func=_map, num_parallel_calls=config['num_parallel_calls'])


    if no_shuffle_and_batching:
        dataset = dataset.unbatch()
    else:
        dataset = dataset.unbatch().batch(config['batch']).shuffle(config['shuffle']).prefetch(
            tf.data.experimental.AUTOTUNE)

    return dataset

def pipeline_melanom(config, dataset_image_path, dataset_label_path, dataset=None, augment=False,
                     training=False, evaluate=False):
    """
    Pipeline of tf.data for importing the data from MELANOM database
    :param config: type dict,config parameter
    :param dataset_image_path: type str: dataset image path
    :param dataset_label_path: type str: dataset label path
    :param dataset: dataset to be used
    :param augment: true if augmentation is to be performed
    :param training: true if training process is to be performed
    :param evaluate: true if evaluation process is to be performed
    :return: dataset: return tf.data.dataset: pipeline dataset
    """
    patch_size = config['patch_size']
    print("patch size: ", patch_size)
    num_classes = config['num_classes']
    num_channels_img = config['channel_img_num']
    patch_shape_with_channels = patch_size + [num_channels_img]
    dim = len(patch_size)
    print("dim: ", dim)
    patches_per_subject = config['patches_per_subject']
    print("patches per subject: ", patches_per_subject)

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

    # Zip dataset of images and labels

    zip_data_path_TFRecordDataset = tf.data.Dataset.zip(
        (list_image_TFRecordDataset[0], list_label_TFRecordDataset[0]))

    ##--------------------------------------------------------------------------------------------------------------
    ## ----------- map functions for the pipeline -------------------------------------------------------

    @tf.function
    def _map(*args):
        """
                Map function of Zip dataset for parsing paths to data
                :param args: args[0] for images TFRecordDataset, args[1] for labels TFRecordDataset
                :return: images_data, labels_data: type list of data tensor in patches
        """

        images_data, images_shape, validation_for_cancer = parser_melanom(args[0])
        labels_data, labels_shape, validation_for_cancer = parser_melanom(args[1])

        # For melanoma dataset, we do not need to pad the data to the max shape form,
        # instead sampled patches are used to be fed in pipeline
        # images_data, labels_data = pad_img_label(config, max_data_size, images_data, images_shape,
        # labels_data, labels_data)

        # class probability ---------------------------------------------------------------------------------
        patch_size_tensor = tf.convert_to_tensor(patch_size)
        class_probabilities = config['class_probabilities']
        _label_ax2_any = []
        class_p = None

        if class_probabilities:
            class_p = class_probabilities / np.sum(class_probabilities)
            max_class_value = len(class_probabilities)

            # for each class value in label
            # compute if any voxel value along axis=2 is equal to the class value
            # this improves the computational efficency of the label sampling
            # signficantly

        if training:
            patchs_imgs, patchs_labels = get_sampled_patches(patch_size_tensor, images_data, labels_data,
                                                             class_p=class_p,
                                                             max_class_value=max_class_value,
                                                             patches_per_subject=patches_per_subject,
                                                             data_shape=images_shape[:-1], dim_patch=dim,
                                                             channel_img=num_channels_img, channel_label=num_classes,
                                                             validation_for_1=validation_for_cancer)

        else:

            patchs_imgs, patchs_labels = get_grid_patch_sample_test(images_data, labels_data,
                                                                    data_shape=images_shape[:-1], patch_size=patch_size,
                                                                    dim_patch=dim,
                                                                    patch_overlap=config['patch_overlap_rate'])

        return patchs_imgs, patchs_labels  # , validation_for_cancer

    @tf.function
    def augmentation_map(patchs_imgs, patchs_labels) -> tf.Tensor:

        """
            Augmentation function of Zip dataset for parsing paths to data for augmentation case.
            Augmentation is performed after patching.
            :param patchs_imgs: image patches from dataset
            :param patchs_labels: label patches from the dataset
            :return: images_data, labels_data: type list of data tensor patches with corresponding augmentation
        """

        ## get corresponding augmentations
        transformation_list = config['augmentation']

        # for this task, just color augmentations were performed
        # for more augmentation transformations, add the new cases

        for transformation in transformation_list:

            if transformation == 'brightness':
                print("performing brightness transformation .................... ")
                patchs_imgs = brightness_transform(patchs_imgs, mu=0.0, sigma=0.3, num_patches=patches_per_subject,
                                                   num_channel=num_channels_img,
                                                   shape_data=patch_shape_with_channels,
                                                   per_channel=True, p_per_channel=1)

            if transformation == 'gamma':
                print("performing gamma transformation .................... ")
                patchs_imgs = gamma_contrast(patchs_imgs, num_patches=patches_per_subject,
                                             num_channel=num_channels_img, gamma_range=(0.7, 1.3),
                                             shape_data=patch_shape_with_channels, invert_image=False,
                                             per_channel=False, retain_stats=False)

            if transformation == 'contrast':
                print("performing contrast transformation .................... ")
                patchs_imgs = contrast_augmentation_transform(patchs_imgs, contrast_range=(0.3, 1.7),
                                                              num_patches=patches_per_subject,
                                                              num_channel=num_channels_img,
                                                              shape_data=patch_shape_with_channels,
                                                              preserve_range=True, per_channel=True, p_per_sample=1)

        return patchs_imgs, patchs_labels

    ## ---------------- end map functions ------------------------------------------------------------------------------
    ##--------------------------------------------------------------------------------------------------------------------

    # Create pipeline and config dataset
    dataset = zip_data_path_TFRecordDataset.map(map_func=_map,
                                                num_parallel_calls=config['num_parallel_calls'])

    if augment:
        dataset = dataset.map(map_func=augmentation_map,
                              num_parallel_calls=config['num_parallel_calls'])


    if not evaluate:

        dataset = dataset.unbatch().batch(config['batch']).shuffle(config['shuffle']).prefetch(4)
    else:

        dataset = dataset.unbatch().batch(config['batch_predict']).shuffle(config['shuffle']).prefetch(4)


    return dataset