import yaml
import tensorflow as tf
from med_io.preprocess_raw_dataset import *
from med_io.read_and_save_datapath import *
from train import *
from evaluate import *
from predict import *
from med_io.calculate_max_shape import *
from util import *
import numpy as np
import random
from Patient.Patient import *
import os
from packaging import version


import argparse


def args_argument():
    parser = argparse.ArgumentParser(prog='MedSeg')

    parser.add_argument('-e', '--exp_name', type=str, default='exp0',
                        help='Name of experiment (subfolder in result_rootdir)')

    parser.add_argument('--preprocess', action="store_true", help='Preprocess the data')
    parser.add_argument('--train', action="store_true", help='Train the model')
    parser.add_argument('--evaluate', action="store_true", help='Evaluate the model')
    parser.add_argument('--predict', action="store_true", help='Predict the model')
    parser.add_argument('--restore', action="store_true", help='Restore the unfinished trained model')
    parser.add_argument('-c', '--config_path', type=str, default='./config/config_default.yaml',
                        help='Configuration file of the project')


    parser.add_argument("--gpu", type=int, default=0, help="Specify the GPU to use")
    parser.add_argument('--gpu_memory', type=float, default=None, help='Set GPU allocation. (in GB) ')
    parser.add_argument('--calculate_max_shape_only', action="store_true",
                        help='Only calculate the max shape of each dataset')
    parser.add_argument('--split_only', action="store_true",
                        help='Only split the whole dataset to train, validation, and test dataset')
    parser.add_argument('--train_epoch', type=int, default=None, help='Modify the train epoch in yaml file')
    parser.add_argument('--filters', type=int, default=None, help='Modify the base filters in yaml file')
    parser.add_argument('--model_name', type=str, default=None, help='Modify the models in yaml file')
    parser.add_argument('--train_batch', type=int, default=None, help='Modify the batch in yaml file')
    parser.add_argument('--dataset', type=str, default=None, help='Modify the dataset in yaml file')

    a = parser.parse_args()
    return a


def main(args):
    # set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    # limit the gpu by allocating the specific GPU memory
    if args.gpu_memory is not None:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * args.gpu_memory)])
            except RuntimeError as e:
                print(e)
    else:  # allocate dynamic growth

        if version.parse(tf.__version__) >= version.parse('2.0'):
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True
            tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
        else:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            tf.set_session(tf.Session(config=config))



    with open(args.config_path, "r") as yaml_file:
        config = yaml.load(yaml_file.read())

        config = convert_yaml_config(config)

    # set random seed to fix the randomness in training
    if config['tensorflow_seed']:
        tf.random.set_seed(config['tensorflow_seed'])
    if config['numpy_seed']:
        np.random.seed(config['numpy_seed'])
    if config['random_seed']:
        random.seed(config['random_seed'])

    if args.exp_name:
        config['exp_name'] = args.exp_name

    if args.train_epoch:
        config['epoch'] = args.train_epoch
    if args.filters:
        config['filters'] = args.filters
    if args.train_batch:
        config['batch'] = args.train_batch
    if args.dataset:
        config['dataset'] = [args.dataset]

    def check_and_write_list_tfrecord():
        for dataset in config['dataset']:

            if (not os.path.isfile(
                    config['dir_list_tfrecord'] + '/' + config['filename_tfrec_pickle'][dataset] + '.pickle') and not
            config['read_body_identification']):
                read_and_save_tfrec_path(config, rootdir=config['rootdir_tfrec'][dataset],
                                         filename_tfrec_pickle=config['filename_tfrec_pickle'][dataset],
                                         dataset=dataset)

            if (not os.path.isfile(
                    config['dir_list_tfrecord'] + '/' + config['filename_tfrec_pickle'][dataset] + '_bi.pickle') and
                    config['read_body_identification']):
                read_and_save_tfrec_path(config, rootdir=config['rootdir_tfrec'][dataset],
                                         filename_tfrec_pickle=config['filename_tfrec_pickle'][dataset],
                                         dataset=dataset)

    # preprocess and convert input to TFRecords
    if args.preprocess:
        preprocess_raw_dataset(config)  # read, convert and store
        calculate_max_shape(config)  # find and dump the max shape
        split(config)  # split into train, validation and test set

    if args.calculate_max_shape_only:
        check_and_write_list_tfrecord()
        calculate_max_shape(config)  # find and dump the max shape
        split(config)  # split into train, validation and test set
    if args.split_only:
        check_and_write_list_tfrecord()
        split(config)  # split into train, validation and test set

    if args.train:  # train the model
        train(config, args.restore)

        print("Training finished for %s" % (config['dir_model_checkpoint'] + os.sep + config['exp_name']))
    if args.evaluate:  # evaluate the metrics of a trained model
        evaluate(config, datasets=config['dataset'])
        print("Evaluation finished for %s" % (config['result_rootdir'] + os.sep + config['exp_name']))

    if args.predict:  # predict and generate output masks of a trained model
        predict(config, datasets=config['dataset'], save_predict_data=config['save_predict_data'])
        print("Prediction finished for %s" % (config['result_rootdir'] + os.sep + config['exp_name']))


if __name__ == '__main__':
    main(args_argument())
