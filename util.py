import tensorflow as tf
import pickle
import os
from sklearn.model_selection import train_test_split
import numpy as np
from models.metrics import get_custom_metrics
from keras.utils.np_utils import to_categorical
import random


def convert_yaml_config(config):
    """
    Convert str in :param config to Obj
    :param config: type dict, config parameters
    :return: config : type dict
    """
    if 'convolution_parameter' in config.keys():
        if 'kernel_regularizer' in config['convolution_parameter'].keys():
            if config['convolution_parameter']['kernel_regularizer'][0] == 'l2':
                config['convolution_parameter']['kernel_regularizer'] = \
                    tf.keras.regularizers.l2(l=config['convolution_parameter']['kernel_regularizer'][1])
            elif config['convolution_parameter']['kernel_regularizer'][0] == 'l1':
                config['convolution_parameter']['kernel_regularizer'] = \
                    tf.keras.regularizers.l1(l=config['convolution_parameter']['kernel_regularizer'][1])

    # function for remove redundant brackets of metrics
    flatten = lambda x: [y for l in x for y in flatten(l)] if type(x) is list else [x]
    tf_metrics = [t_metric for t_metric in config['tensorflow_metrics']]

    custom_metrics = flatten(config['custom_metrics'])
    config['metrics'] = tf_metrics + custom_metrics

    return config


def split(config):
    """
    Split the dataset paths into training, validation, and test dataset paths (patient leave-out approach)."
    :param config: type dict: config parameter
    :return:
    """

    for dt in config['dataset']:
        if config['read_body_identification']:
            filename = config['dir_list_tfrecord'] + '/' + config['filename_tfrec_pickle'][dt] + '_bi.pickle'
        else:
            filename = config['dir_list_tfrecord'] + '/' + config['filename_tfrec_pickle'][dt] + '.pickle'
        with open(filename, 'rb') as fp:
            dataset_path = pickle.load(fp)
        # pickle files dict patterns must be:
        # {'image': tfrecord list of images path,'label':  tfrecord list of labels path }
        dataset_image_path = dataset_path['image']  # type list of str
        dataset_label_path = dataset_path['label']  # type list of str
        # dataset_info_path = dataset_path['info']  # type list of str

        # join images paths with info paths and then split them after the splitting in training and dataset
        # dataset_image_info_path = list(zip(dataset_image_path, dataset_info_path))

        dict_dataset = dict()
        if config['test_subject'][dt] is not None:
            path_test_img, path_test_label = [dataset_image_path[idx] for idx in config['test_subject'][dt]], [
                dataset_label_path[idx] for idx in config['test_subject'][dt]]
            path_train_val_img, path_train_val_label = [dataset_image_path[idx] for idx in
                                                        range(len(dataset_image_path)) if
                                                        not idx in config['test_subject'][dt]], [dataset_label_path[idx]
                                                                                                 for idx in range(
                    len(dataset_label_path)) if not idx in config['test_subject'][dt]]
        else:
            path_train_val_img, path_test_img, path_train_val_label, path_test_label = train_test_split \
                (dataset_image_path, dataset_label_path, test_size=config['ratio_test_to_train_val'],
                 random_state=config['seed_random_split'])
            # path_train_val_img_info, path_test_img_info, path_train_val_label, path_test_label = train_test_split \
            # (dataset_image_info_path, dataset_label_path, test_size=config['ratio_test_to_train_val'],
            # random_state=config['seed_random_split'])

        dict_dataset['path_test_img'], dict_dataset['path_test_label'] = path_test_img, path_test_label
        dict_dataset['path_train_val_img'], dict_dataset[
            'path_train_val_label'] = path_train_val_img, path_train_val_label

        path_train_img, path_val_img, path_train_label, path_val_label = \
            train_test_split(path_train_val_img, path_train_val_label, test_size=config['ratio_val_to_train'],
                             random_state=config['seed_random_split'])

        # path_train_img_info, path_val_img_info, path_train_label, path_val_label = \
        #    train_test_split(path_train_val_img_info, path_train_val_label, test_size=config['ratio_val_to_train'],
        #                     random_state=config['seed_random_split'])

        # unzip the paths of images and info
        # path_train_img, path_train_info = [list(t) for t in zip(*path_train_img_info)]
        # path_val_img, path_val_info = [list(t) for t in zip(*path_val_img_info)]
        # path_test_img, path_test_info = [list(t) for t in zip(*path_test_img_info)]

        # dict_dataset['path_test_img'], dict_dataset['path_test_label'] = path_test_img, path_test_label

        # dict_dataset['path_train_val_img_info'], dict_dataset[
        #    'path_train_val_label'] = path_train_val_img_info, path_train_val_label

        dict_dataset['path_train_img'], dict_dataset['path_train_label'] = path_train_img, path_train_label
        dict_dataset['path_val_img'], dict_dataset['path_val_label'] = path_val_img, path_val_label
        # dict_dataset['path_train_info'], dict_dataset['path_val_info'] = path_train_info, path_val_info
        # dict_dataset['path_test_info'] = path_test_info

        if not os.path.exists(config['dir_dataset_info']): os.makedirs(config['dir_dataset_info'])
        if config['read_body_identification']:
            pickle_filename = config['dir_dataset_info'] + '/split_paths_' + dt + '_bi.pickle'
        else:
            pickle_filename = config['dir_dataset_info'] + '/split_paths_' + dt + '.pickle'

        with open(pickle_filename, 'wb') as fp:
            pickle.dump(dict_dataset, fp, protocol=pickle.HIGHEST_PROTOCOL)
            print('Sucessfully save split paths files.')


def convert_tf_optimizer(config):
    """
    parse str in the yaml file -> tensorflow optimizer functions
    :param config: type dict: config parameter
    :return: function in tf.keras.optimizers
    """
    optimizer_name = config['optimizer']['name']
    if optimizer_name == 'SGD':
        return tf.keras.optimizers.SGD(**config['optimizer']['args'])
    elif optimizer_name == 'Nadam':
        return tf.keras.optimizers.Nadam(**config['optimizer']['args'])
    elif optimizer_name == 'Adadelta':
        return tf.keras.optimizers.Adadelta(**config['optimizer']['args'])
    elif optimizer_name == 'Adagrad':
        return tf.keras.optimizers.Adagrad(**config['optimizer']['args'])
    elif optimizer_name == 'Adam':
        return tf.keras.optimizers.Adam(**config['optimizer']['args'])
    elif optimizer_name == 'Adamax':
        return tf.keras.optimizers.Adamax(**config['optimizer']['args'])
    elif optimizer_name == 'Ftrl':
        return tf.keras.optimizers.Ftrl(**config['optimizer']['args'])
    elif optimizer_name == 'RMSprop':
        return tf.keras.optimizers.RMSprop(**config['optimizer']['args'])


def convert_integers_to_onehot(img, num_classes=3):
    # if some values in img > num_classes-1=> error
    return to_categorical(img, num_classes=num_classes)


def convert_onehot_to_integers(img, axis=-1):
    """
    Convert onehot encoding to integer
    """
    return np.argmax(img, axis=axis)


def convert_onehot_to_integers_add_bg(img, axis=-1):
    """
    Convert onehot encoding to integer, considering background
    """
    shape_img = list(img.shape)
    shape_img[-1] += 1
    new_img = np.zeros(tuple(shape_img))
    new_img[..., 1:] = img
    return np.argmax(new_img, axis=axis)


def get_thresholds(decision_map, n_classes=6, row_dim=1):
    """
    Calculates thresholds (line) of predicted decision map
    input:  decision_map: type ndarray (size_m* size_n), each element represents chass
            n_classes: number of classes to calculate thresholds from
    """
    row_max = np.zeros(decision_map.shape[0],
                       dtype='int32')  # calculate max value( highest frequency class) of each decision map row
    for i in range(decision_map.shape[0]):
        class_cnt = np.zeros(n_classes, dtype='int32')  # calculate sum of each elem in decision_map row
        for j in range(decision_map.shape[1]):
            class_cnt[decision_map[i, j]] += 1
        row_max[i] = np.argmax(class_cnt)

    thresholds = np.zeros(n_classes - 1, dtype='int32')
    idx = 0

    for i in range(decision_map.shape[0] - 2):
        # if row_max[i + 1] == row_max[i + 2] --> this row is the threshold line of this class
        if row_max[i] != row_max[i + 1] and row_max[i + 1] == row_max[i + 2]:
            thresholds[idx] = i + 1
            idx += 1
            if idx == n_classes - 1:
                return thresholds
    return thresholds
