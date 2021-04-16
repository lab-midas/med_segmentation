print("hello world")

import tensorflow as tf
from med_io.pipeline import *
import pickle
import yaml
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
def train_config_setting(config, dataset):
    """
    Configuring parameter for training
    :param config: type dict: config parameter
    :param dataset: type str: dataset  name
    :return: config: type dict: config parameter
    """

    print(config['dir_dataset_info'])
    # Load max shape & channels of images and labels.
    if config['read_body_identification']:
        filename_max_shape = config['dir_dataset_info'] + '/max_shape_' + dataset + '_bi.pickle'
    else:
        filename_max_shape = config['dir_dataset_info'] + '/max_shape_' + dataset + '.pickle'
    with open(filename_max_shape, 'rb') as fp:
        config['max_shape'] = pickle.load(fp)
    # Get the amount of input and output channel
    # config[channel_img]: channel amount of model input, config[channel_label]: channel amount of model output
    config['channel_img_num'], config['channel_label_num'] = config['max_shape']['image'][-1], \
                                                             config['max_shape']['label'][
                                                                 -1]
    if config['input_channel'][dataset] is not None:
        config['channel_img_num'] = len(config['input_channel'][dataset])

    if not config['read_body_identification']:
        if config['output_channel'][dataset] is not None:
            config['channel_label_num'] = len(config['output_channel'][dataset])

        # output channel+1 if the model output background channel (if the stored labels have no background channels)
        if config['model_add_background_output']:
            config['channel_label_num'] += 1

    print('channel_img,', config['channel_img_num'], 'channel_label,', config['channel_label_num'])
    return config

config_path = '../config/config_default.yaml'

with open(config_path, "r") as yaml_file:
    config = yaml.load(yaml_file.read(), Loader=yaml.FullLoader)
    config = convert_yaml_config(config)

dataset = 'MELANOM'#config['dataset']
print(dataset)
print(type(dataset))

config = train_config_setting(config, dataset)
split_filename = config['dir_dataset_info'] + '/split_paths_' + dataset + '.pickle'
with open(split_filename, 'rb') as fp:
    paths = pickle.load(fp)

ds_train = pipeline(config, paths['path_train_img'], paths['path_train_label'], dataset=dataset)

#ds_validation = pipeline(config, paths_val_img, paths_val_label, dataset=dataset)
print("Size of dataset training: ", len(ds_train))
#print("Size of dataset validation: ", len(ds_validation))