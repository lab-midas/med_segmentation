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
from tests.test_gamma import *
from tests.test_patch import *
import pickle
from keras.utils.np_utils import to_categorical

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
    patches_per_subject = config['patches_per_subject']
    print("patches per subject: ", patches_per_subject)
    # Read max data size of this dataset
    if not config['read_body_identification']:
        max_data_size = [max(config['max_shape']['image'][i], config['max_shape']['label'][i]) for i in range(dim)]
    else:
        max_data_size = [config['max_shape']['image'][i] for i in range(dim)]

    num_classes = config['num_classes']
    num_channels_img = config['channel_img']

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
    def _map_data_augmentation(*args):
        ####----------------------------------
        ##augmentation
        # this does not correspond to pytorch architecture
        ## augmentation is performed after patching
        ## augmentations used are contrast and brightness change: gamma, brightness, contrast
        ## further details implemented in PyTorch archictecture
        ###------------------------------------

        """
                Map function of Zip dataset for parsing paths to data for augmentation case
                :param args: args[0] for images TFRecordDataset, args[1] for labels TFRecordDataset
                :return: images_data, labels_data: type list of data tensor
        """

        images_data, images_shape = parser(args[0])
        labels_data, labels_shape = parser(args[1])
        print("images_shape: ", images_shape)
        print("labels_shape: ", labels_shape)

        # For melanoma dataset, we do not need to pad the data to the max shape form,
        # instead sampled patches are used to be fed in pipeline
        #images_data, labels_data = pad_img_label(config, max_data_size, images_data, images_shape,
                                                 #labels_data, labels_data)

        # class probability ---------------------------------------------------------------------------------
        class_probabilities = config['class_probabilities']
        pos = None
        _label_ax2_any = []
        selected_class = 0
        if class_probabilities is not None:
            class_probabilities = class_probabilities / np.sum(class_probabilities)
            max_class_value = len(class_probabilities)

            for idx in range(len(config['num_labels'])):
                _label_ax2_any.append([np.any(labels[idx][..., -1] == c, axis=2)

            for c in range(max_class_value)])

            selected_class = np.random.choice(range(len(class_probabilities)),
                                              p=class_probabilities)

            if selected_class > 0:
                pos = get_labeled_position(lbls[-1], selected_class,
                                           label_any=self._label_ax2_any[idx][selected_class])

        # get valid indices of a random patch containing the specified position
        index_ini, index_fin = get_random_patch_indices(patch_size, imgs.shape[1:], pos=pos)

        # ----------------------------------------------------------------------------------------------------


        # Patch the data
        patchs_imgs, patchs_labels, index_list = get_patches_data(max_data_size, patch_size, images_data,
                                                                  labels_data,
                                                                  patches_indices, slice_channel_img=input_slice,
                                                                  slice_channel_label=output_slice,
                                                                  output_patch_size=config['model_output_size'])

        ##---------- augmentation at image level------------------------------------------------

        transformation_list = config['augmentation']

        for transform in transformation_list:
            print("transformation: ", transform)

        for transformation in transformation_list:

            if transformation == 'gamma':
                images_data = gamma_contrast(patchs_imgs, max_data_size, num_classes, gamma_range=(0.7, 1.3),
                                             num_patches=len(patches_indices), num_channel=num_channels_img,
                                             shape_data=max_data_size)
            if transformation == 'brightness':
                images_data = brightness_transform(patchs_imgs, mu=0.0, sigma=0.3,
                                                   num_patches=len(patches_indices), num_channel=num_channels_img,
                                                   shape_data=max_data_size)

            if transformation == 'contrast':
                images_data = contrast_augmentation_transform(patchs_imgs, contrast_range=(0.3, 1.7),
                                                              num_patches=len(patches_indices),
                                                              num_channel=num_channels_img, shape_data=max_data_size)

        ## List regularize
        index_list = index_list / (np.array(max_data_size) + 1e-16)
        if config['feed_pos']:
            return (patchs_imgs, index_list), patchs_labels
        else:
            return patchs_imgs, patchs_labels

    @tf.function
    def _map(*args):

        """
        Map function of Zip dataset for parsing paths to data
        :param args: args[0] for images TFRecordDataset, args[1] for labels TFRecordDataset
        :return: images_data, labels_data: type list of data tensor
        """
        images_data, images_shape = parser(args[0])
        labels_data, labels_shape = parser(args[1])
        #print("images_shape: ", images_shape)
        #print("labels_shape: ", labels_shape)

        # Pad the data
        images_data, labels_data = pad_img_label(config, max_data_size, images_data, images_shape, labels_data,
                                                 labels_shape)

        # class probability ---------------------------------------------------------------------------------
        class_probabilities = config['class_probabilities']
        pos = None
        selected_class = 0
        if class_probabilities is not None:
            class_probabilities = class_probabilities / np.sum(class_probabilities)
            max_class_value = len(class_probabilities)

            for idx in range(len(self.labels)):
                self._label_ax2_any.append([np.any(self.labels[idx][-1, ...] == c, axis=2)
                                            for c in range(max_class_value)])

            selected_class = np.random.choice(range(len(class_probabilities)),
                                              p=class_probabilities)

            if selected_class > 0:
                pos = get_labeled_position(lbls[-1], selected_class,
                                           label_any=self._label_ax2_any[idx][selected_class])

        # get valid indices of a random patch containing the specified position
        index_ini, index_fin = get_random_patch_indices(self.patch_size, imgs.shape[1:], pos=pos)

        cropped_imgs = imgs[:, index_ini[0]:index_fin[0],
                       index_ini[1]:index_fin[1],
                       index_ini[2]:index_fin[2]].astype(np.float32)

        cropped_lbls = lbls[:, index_ini[0]:index_fin[0],
                       index_ini[1]:index_fin[1],
                       index_ini[2]:index_fin[2]].astype(np.uint8)

        #----------------------------------------------------------------------------------------------------

        # Patch the data
        patchs_imgs, patchs_labels, index_list = get_patches_data(max_data_size, patch_size, images_data,
                                                                  labels_data,
                                                                  patches_indices, slice_channel_img=input_slice,
                                                                  slice_channel_label=output_slice,
                                                                  output_patch_size=config['model_output_size'])
        #print("patch type: ", type(patchs_imgs))
        #print("patch shape: ", patchs_imgs[0].shape)
        #print(patchs_imgs)

        ## List regularize
        index_list = index_list / (np.array(max_data_size) + 1e-16)
        if config['feed_pos']:
            return (patchs_imgs, index_list), patchs_labels
        else:
            return patchs_imgs, patchs_labels

    # Create pipeline and config dataset.
    # dataset = zip_data_path_TFRecordDataset.map(map_func=_map, num_parallel_calls=config['num_parallel_calls'])
    if augment: # for training dataset
        dataset = zip_data_path_TFRecordDataset.map(map_func=_map_data_augmentation,
                                                    num_parallel_calls=config['num_parallel_calls'])

    else:
        dataset = zip_data_path_TFRecordDataset.map(map_func=_map,
                                                    num_parallel_calls=config['num_parallel_calls'])

    #dataset = dataset.unbatch().batch(8)
    #print(dataset.element_spec)

    #for elem in dataset.take(1):
    #    img_elem = elem[0]
    #    mask_elem = elem[1]
    #    print("image size: ", elem[0].shape)
    #    print("image type: ", type(elem[0]))
    #    print("label size: ", elem[1].shape)
    #    print("label type: ", type(elem[1]))
    #    #------------------------------------------------
    #path_test = './tests/test_img'
    #if not os.path.exists(path_test): os.makedirs(path_test)
    #with open(path_test + '/' + 'elem.pickle', 'wb') as fp:
    #    pickle.dump({'image': img_elem, 'mask': mask_elem}, fp, protocol=pickle.HIGHEST_PROTOCOL)
        #--------------------------------------------------
        #slice_cut = 'a'
        #slice_x_list = [10, 25, 50, 65, 75, 90]
        #for batch_x in range(8):
            #for slice_x in slice_x_list:
                #test_patch(elem[0], slice_x, slice_cut, batch_x)


    #print("dataset type: ", type(dataset))

    #dataset = dataset.unbatch().shuffle(500).batch(config['batch']).prefetch(tf.data.experimental.AUTOTUNE)
    dataset = dataset.unbatch().batch(config['batch']).shuffle(config['shuffle']).prefetch(8)
    # tf.data.experimental.AUTOTUNE)

    #print("dataset type: ", type(dataset))

    return dataset
