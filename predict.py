import tensorflow as tf
from med_io.pipeline import *
from models.ModelSet import *
from models.Premodel_Set import *

from models.loss_function import *
from models.metrics import *
from util import *
import pickle
import os
from util import *
from med_io.parser_tfrec import parser
from med_io.get_pad_and_patch import *
from plot.plot_figure import *
from plot.plot_config import *
from models.Premodel_Custom_Class import *
from models.load_model import load_model_file
from med_io.read_mat import read_mat_file
from med_io.read_dicom import read_dicom_dir
from med_io.read_nii import read_nii_path
from predict_data_processing.nifti_process import read_nifti
import scipy.io as sio
import numpy as np


def predict(config, datasets=None, save_predict_data=False, name_ID=None):
    """
    Predict the test data based on the saved model
    :param config: type dict: config parameter
    :param datasets: type list of str, names of dataset
    :param save_predict_data: type bool, names of dataset
    :param save_predict_data: type bool, names of dataset
    :return:
    """
    if datasets is None:
        datasets = config['dataset']
    if config['load_predict_from_tfrecords']:
        #  If predict data is in tfrecords format
        for dataset in datasets:
            if not name_ID:
                # Load path of predict dataset.
                with open(config['dir_dataset_info'] + '/split_paths_' + dataset + '.pickle', 'rb') as fp:
                    split_path = pickle.load(fp)
                    dataset_image_path = split_path['path_test_img']
                    dataset_label_path = split_path['path_test_label']

            else:
                # load tfrecords by a single name_ID
                dataset_image_path = [[config['rootdir_tfrec'][dataset] + '/' + name_ID + '/image/image.tfrecords']]
                dataset_label_path = [[config['rootdir_tfrec'][dataset] + '/' + name_ID + '/label/label.tfrecords']]

            config = channel_config(config, dataset)
            # Reformat data path list: [[path1],[path2], ...] ->[[path1, path2, ...]]
            data_path_image_list = [t[i] for t in dataset_image_path for i in range(len(dataset_image_path[0]))]
            data_path_label_list = [t[i] for t in dataset_label_path for i in range(len(dataset_label_path[0]))]
            list_image_TFRecordDataset = [tf.data.TFRecordDataset(i) for i in data_path_image_list]
            list_label_TFRecordDataset = [tf.data.TFRecordDataset(i) for i in data_path_label_list]

            # Choose and create the model which is the same with the saved model.
            model = load_model_file(config, dataset)

            collect_predict, collect_label = [], []

            for index, (image_TFRecordDataset, label_TFRecordDataset, data_path_image) in \
                    enumerate(zip(list_image_TFRecordDataset, list_label_TFRecordDataset, data_path_image_list)):
                dataset_image = image_TFRecordDataset.map(parser)
                dataset_label = label_TFRecordDataset.map(parser)

                # Get the image data from tfrecords
                # elem[0]= data, elem[1]= data shape
                img_data = [elem[0].numpy() for elem in dataset_image][0]
                label_data_onehot = [elem[0].numpy() for elem in dataset_label][0]
                img_data, label_data_onehot = image_transform(config, img_data, label_data_onehot)
                # Patch the image
                patch_imgs, indice_list = patch_image(config, img_data)
                print('76',patch_imgs.shape)
                predict_img = predict_image(config, dataset, model, patch_imgs, indice_list)
                predict_img, label_data_onehot = select_output_channel(config, dataset, predict_img, label_data_onehot)
                predict_img_integers, predict_img_onehot, label_data_integers, label_data_onehot = convert_result(
                    config, predict_img, label_data_onehot)

                # Get name_ID from the data path
                # The data path must have the specified format which is generated from  med_io/preprocess_raw_dataset.py
                name_ID = data_path_image.replace('\\', '/').split('/')[-3]

                # Get data of one patient for plot
                dict_data = {'predict_integers': predict_img_integers,
                             'predict_onehot': predict_img_onehot,
                             'label_integers': label_data_integers,
                             'label_onehot': label_data_onehot,
                             'original_image': img_data,
                             'without_mask': np.zeros(predict_img_integers.shape)}
                if config['plot_figure']:
                    # Plot figure based on the single patient
                    plot_figures_single(config, dict_data, dataset=dataset, name_ID=name_ID)
                if save_predict_data:
                    save_img_mat(config, dataset, name_ID, 'predict_image', predict_img_integers)
                # Collect the image for plot.
                collect_predict.append(predict_img_onehot)
                collect_label.append(label_data_onehot)
            # list_images_series: collect  predict data and label data
            list_images_series = {'predict': collect_predict, 'label': collect_label}
            if config['plot_figure']:
                plot_figures_dataset(config, list_images_series, dataset=dataset)

            print('Predict data ', dataset, 'is finished.')
    else:
        # Load dataset not from tfrecords. e.g. from nifti
        if datasets is None or datasets == []:
            datasets = ['New_predict_image']
        for dataset in datasets:
            config = channel_config(config, dataset)
            # Choose and create the model which is the same with the saved model.
            model = load_model_file(config, dataset)
            collect_predict, collect_label = [], []
            data_dir_img = config['predict_data_dir_img']  # get the image dir
            # If predict dataset has label
            if config['predict_load_label']:
                data_dir_label = config['predict_data_dir_label']
                for index, (dir_name_img, dir_name_label) in \
                        enumerate((os.listdir(data_dir_img), os.listdir(data_dir_label))):
                    data_path_img = os.path.join(config['predict_data_dir_img'], dir_name_img).replace('\\', '/')
                    data_path_label = os.path.join(config['predict_data_dir_label'], dir_name_label).replace('\\', '/')
                    img_data, label_data = read_predict_file(config, data_path_img, data_path_label)
                    img_data, label_data_onehot = image_transform(config, img_data, label_data_onehot=label_data)
                    # Patch the image
                    patch_imgs, indice_list = patch_image(config, img_data)
                    predict_img = predict_image(config, dataset, model, patch_imgs, indice_list)
                    predict_img, label_data_onehot = select_output_channel(config, dataset, predict_img,
                                                                           label_data_onehot)
                    predict_img_integers, predict_img_onehot, label_data_integers, label_data_onehot \
                        = convert_result(config, predict_img, label_data_onehot=label_data)
                    name_ID = dir_name_img
                    # Get data of one patient for plot
                    dict_data = {'predict_integers': predict_img_integers,
                                 'predict_onehot': predict_img_onehot,
                                 'label_integers': label_data_integers,
                                 'label_onehot': label_data_onehot,
                                 'original_image': img_data,
                                 'without_mask': np.zeros(predict_img_integers.shape)}
                    if config['plot_figure']:
                        plot_figures_single(config, dict_data, dataset=dataset, name_id=name_ID)
                    if save_predict_data:
                        save_img_mat(config, dataset, name_ID, 'predict_image', predict_img_integers)
                    # Collect the image for plot.
                    collect_predict.append(predict_img_onehot)
                    collect_label.append(label_data_onehot)
            else:
                # Load dataset not from tfrecords. e.g. from nifti, and have no labels
                for name_ID in os.listdir(data_dir_img):
                    data_path_img = os.path.join(data_dir_img, name_ID).replace('\\', '/')
                    img_data = read_predict_file(config, data_path_img, name_ID=name_ID)
                    img_data = image_transform(config, img_data)
                    patch_imgs, indice_list = patch_image(config, img_data)
                    predict_img = predict_image(config, dataset, model, patch_imgs, indice_list,
                                                img_data_shape=img_data.shape)
                    predict_img = select_output_channel(config, dataset, predict_img)
                    predict_img_integers, predict_img_onehot = convert_result(config, predict_img)
                    dict_data = {'predict_integers': predict_img_integers,
                                 'predict_onehot': predict_img_onehot,
                                 'label_integers': None,
                                 'label_onehot': None,
                                 'original_image': img_data,
                                 'without_mask': np.zeros(predict_img_integers.shape)}
                    # Plot single figure
                    if config['plot_figure']:
                        plot_figures_single(config, dict_data, dataset=dataset, name_id=name_ID)
                    if save_predict_data:
                        save_img_mat(config, dataset, name_ID, 'predict_image', predict_img_integers)
                    # Collect the image for plot.
                    collect_predict.append(predict_img_onehot)
                    collect_label.append(None)
                # dict collect
                list_images_series = {'predict': collect_predict, 'label': collect_label}
                if config['plot_figure']:
                    plot_figures_dataset(config, list_images_series, dataset=dataset)
                print('Predict data', dataset, 'is finished.')


def channel_config(config, dataset, evaluate=False):
    """
    Set the channel of config files
    :param config: type dict: config parameter
    :param datasets: type list of str, names of dataset
    :return: config: type dict: config parameter
    """
    if config['load_predict_from_tfrecords'] or evaluate:
        # Load max shape & channels of images and labels.
        if not os.path.exists(config['dir_dataset_info'] + '/split_paths_' + dataset + '.pickle'):
            raise FileNotFoundError('Paths of dataset   `config[dir_dataset_info]/split_paths.pickle`are not found! ')
        with open(config['dir_dataset_info'] + '/max_shape_' + dataset + '.pickle', 'rb') as fp:
            config['max_shape'] = pickle.load(fp)

        # Read num of channels of images and labels from the file 'max_shape.pickle'.
        config['channel_img_num'], config['channel_label_num'] = config['max_shape']['image'][-1],\
                                                                 config['max_shape']['label'][-1]
    # Get the total num of input and output channel of the model
    if config['input_channel'][dataset] is not None:
        config['channel_img_num'] = len(config['input_channel'][dataset])
    if config['output_channel'][dataset] is not None:
        config['channel_label_num'] = len(config['output_channel'][dataset])
    print(config['channel_img_num'], config['channel_label_num'])
    if (not config['load_predict_from_tfrecords']) and (not evaluate) and (
            (not config['input_channel'][dataset]) or (not config['output_channel'][dataset])):
        raise ValueError('channel_label must be valued.')
    # If add background channel in model in prediction
    if config['model_add_background_output']:
        config['channel_label_num'] += 1
    print('Input channels amount: ', config['channel_img_num'], 'Output channels amount:',
          config['channel_label_num'])

    return config


def image_transform(config, img_data, label_data_onehot=None):
    """
    Tranform the input image or label data
    :param config:  type dict: config parameter
    :param img_data: type ndarray, image input data
    :param label_data_onehot: type ndarray,  one hot label of the :param img_data
    :return: img_data, label_data_onehot
    """
    if config['predict_image_scale']:
        img_data = img_data * config['predict_image_scale']
    if config['transpose_permute'] is not None:
        img_data = np.transpose(img_data, tuple(config['transpose_permute']))
        if label_data_onehot is not None:
            label_data_onehot = np.transpose(label_data_onehot, tuple(config['transpose_permute']))
    if config['flip_axis'] is not None:
        for ax in config['flip_axis']:
            img_data = np.flip(img_data, axis=ax)
            if label_data_onehot is not None:
                label_data_onehot = np.flip(label_data_onehot, axis=ax)
    if label_data_onehot is not None:
        return img_data, label_data_onehot
    else:
        return img_data


def patch_image(config, img_data):
    """Patch the image"""
    patch_imgs, indice_list = get_predict_patches_index(img_data, config['patch_size'],
                                                        overlap_rate=config[
                                                            'predict_patch_overlap_rate'],
                                                        start=None,
                                                        output_patch_size=config['model_output_size'])
    return np.float32(np.array(patch_imgs)), indice_list


def predict_image(config, dataset, model, patch_imgs, indice_list, img_data_shape=None):
    """
    Predict the image from the model
    :param config: type dict: config parameter
    :param dataset: type str, name of dataset
    :param model: type tf.keras.Model, the Model of prediction
    :param patch_imgs: type list of ndarray, the output
    :param indice_list:
    :param img_data_shape:
    :return:predict_img
    """
    # Select the input channels, which are correspondent to model input channels
    if config['input_channel'][dataset] is not None:
        patch_imgs = patch_imgs[..., config['input_channel'][dataset]]
    indice_list_model = indice_list
    if config['regularize_indice_list']['max_shape']:
        indice_list_model = np.float32(np.array(indice_list)) / np.array(config['max_shape']['image'])[..., 0]
    elif config['regularize_indice_list']['image_shape']:
        indice_list_model = np.float32(np.array(indice_list)) / np.array(
            img_data_shape)[:-1]
    elif config['regularize_indice_list']['custom_specified']:
        indice_list_model = np.float32(np.array(indice_list)) / np.array(
            config['regularize_indice_list']['custom_specified'])
    # Predict the test data by given trained model
    try:
        predict_patch_imgs = model.predict(x=(patch_imgs, indice_list_model), batch_size=1, verbose=1)
    except:
        print('Predict by model with load_weights_only=True Failed, Try rebuild model with load_weights_only=False...')
        config['load_weights_only']= False
        model = load_model_file(config, dataset)
        predict_patch_imgs = model.predict(x=(patch_imgs, indice_list_model), batch_size=1, verbose=1)
    # patch images-> whole image
    predict_img = unpatch_predict_image(predict_patch_imgs, indice_list, config['patch_size'],
                                        output_patch_size=config['model_output_size'],
                                        set_zero_by_threshold=config['set_zero_by_threshold'],
                                        threshold=config['unpatch_start_threshold'])
    # Adjust model output channel order
    if config['predict_output_channel_order']:
        channel_stack = []
        for j in config['predict_output_channel_order']:
            channel_stack.append(predict_img[..., j])
        predict_img = np.stack(tuple(channel_stack), axis=-1)
    return predict_img


def select_output_channel(config, dataset, predict_img, label_data_onehot=None):
    """

    :param config: type dict: config parameter
    :param dataset: type str, name of dataset
    :param predict_img: type ndarray, predict image
    :param label_data_onehot: type ndarray, onehot label image
    :return: predict_img, label_data_onehot
    """
    # one hot output -> integers map
    if config['select_premodel_output_channel'] is not None:
        predict_img = predict_img[..., config['select_premodel_output_channel']]
    if label_data_onehot is not None:
        # Select the label channels, which are correspondent to model output channels
        if config['output_channel'][dataset] is not None:
            label_data_onehot = label_data_onehot[..., config['output_channel'][dataset]]
    if label_data_onehot is not None:
        return predict_img, label_data_onehot
    else:
        return predict_img


def convert_result(config, predict_img, label_data_onehot=None, predict_class_num=None):
    """
    Convert integers images to onehot images, or onehot to integers.
    :param config: type dict: config parameter
    :param predict_img: type ndarray, predict image
    :param label_data_onehot: type ndarray, onehot label image
    :param predict_class_num:  type int, number of class for predict image. None if use the number of label channel
    :return: predict_img_integers, predict_img_onehot, label_data_integers, label_data_onehot
    """

    if config['predict_result_add_background_output']:
        predict_img_integers = convert_onehot_to_integers_add_bg(predict_img)
    else:
        predict_img_integers = convert_onehot_to_integers(predict_img)
    if predict_class_num is None:
        predict_img_onehot = convert_integers_to_onehot(predict_img_integers, num_classes=predict_img.shape[-1])
    else:
        predict_img_onehot = convert_integers_to_onehot(predict_img_integers, num_classes=predict_class_num)
    # add background on label
    channel_label_num = config['channel_label_num']
    if label_data_onehot is not None:
        if config['label_add_background_output']:
            label_data_integers = convert_onehot_to_integers_add_bg(label_data_onehot)
            channel_label_num += 1
        else:
            label_data_integers = convert_onehot_to_integers(label_data_onehot)
        # recreate one hot predict image and label from the integers maps
        label_data_onehot = convert_integers_to_onehot(label_data_integers,
                                                       num_classes=channel_label_num)
        if (not predict_class_num) and (predict_img.shape[-1] != config['channel_label_num']):
            print('Warning! The channels of predict image and label are not equal! Predict:',
                  predict_img.shape[-1], 'Label:', channel_label_num)
        return predict_img_integers, predict_img_onehot, label_data_integers, label_data_onehot
    else:
        return predict_img_integers, predict_img_onehot


def save_img_mat(config, dataset, name_ID, item, data):
    # Config experiment
    save_predict_data_dir = config['result_rootdir'] + '/' + config['exp_name'] + '/' + config[
        'model'] + '/predict_result/' + dataset + '/' + name_ID
    if not os.path.exists(save_predict_data_dir): os.makedirs(save_predict_data_dir)
    save_path = save_predict_data_dir + '/' + 'predict_' + config[
        'model'] + '_' + dataset + '_' + name_ID + '.mat'

    # Save the predict data in .mat file.
    sio.savemat(save_path, {item: data})


def read_predict_file(config, data_path_img=None, data_path_label=None, name_ID=None):
    """
    Read predict data file without using tfrecords
    :param config:  type dict: config parameter
    :param data_path_img: type str: path of image data file (for dicom: dir of all dicom files for one patient, others: path of one patient file )
    :param data_path_label: type str: path of label data file
    :param name_ID: type str: name of patient
    :return: img_data, label_data
    """
    img_data, label_data = None, None

    if data_path_img:
        datatype = config['predict_img_datatype']
        if datatype == 'dicom':
            img_data = read_dicom_dir(data_path_img)
        elif datatype == 'mat':
            img_data, _, _ = read_mat_file(data_path_img, read_label=None, read_info=None)
        elif datatype == 'nii':
            img_data = read_nifti(config, name_ID)
            pass
    if data_path_label:
        datatype = config['predict_label_datatype']
        if datatype == 'nii':
            label_data = read_dicom_dir(data_path_img)
        elif datatype == 'mat':
            _, label_data, _ = read_mat_file(data_path_img)
        else:
            pass
    if data_path_img and data_path_label:
        return img_data, label_data
    elif data_path_img:
        return img_data
    elif data_path_label:
        return label_data
    else:
        return
