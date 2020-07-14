import tensorflow as tf
import pickle
from med_io.parser_tfrec import parser
import numpy as np
import os


def calculate_max_shape(config):
    """
        Calculate the maxmimum shape of the images in the database
        Note:Image and label must have the same size!
        :param config: config file specifying the databases to use
        :return: pickle dump of max shape into database info.
    """

    for dataset in config['dataset']:
        if config['read_body_identification']:
            pickle_path = config['dir_list_tfrecord']+'/list_' + dataset +'_bi'
        else:
            pickle_path = config['dir_list_tfrecord'] + '/list_' + dataset

        with open(pickle_path + '.pickle', 'rb') as fp:
            dataset_path = pickle.load(fp)

        # pickle files dict patterns must be:
        # {'image': tfrecord list of images path,'label':  tfrecord list of labels path }
        dataset_image_path = dataset_path['image']
        dataset_label_path = dataset_path['label']

        data_path_image_list = [t[i] for t in dataset_image_path for i in range(len(dataset_image_path[0]))]
        data_path_label_list = [t[i] for t in dataset_label_path for i in range(len(dataset_label_path[0]))]

        # Create TFRecordDataset for each image and label path
        list_image_TFRecordDataset = tf.data.TFRecordDataset(data_path_image_list)
        list_label_TFRecordDataset = tf.data.TFRecordDataset(data_path_label_list)

        dataset_ = list_image_TFRecordDataset.map(parser)
        img_shape = np.array([elem[1].numpy() for elem in dataset_]) # 0 for data, 1 for data shape
        print('Now calculating max shape of ',dataset)
        img_shape = [max(img_shape[:, i]) for i in range(img_shape.shape[1])]

        dataset_ = list_label_TFRecordDataset.map(parser)
        label_shape = np.array([elem[1].numpy() for elem in dataset_])
        label_shape = [max(label_shape[:, i]) for i in range(label_shape.shape[1])]
        print('Data ', dataset, ': max image shape:', img_shape, 'max label shape:', label_shape)
        dictionary={"image": img_shape, "label": label_shape,"dataset": dataset}
        if not os.path.exists(config['dir_dataset_info']): os.makedirs(config['dir_dataset_info'])

        if config['read_body_identification']:
            pickle_filename = config['dir_dataset_info']+'/max_shape_'+dataset+'_bi.pickle'
        else:
            pickle_filename = config['dir_dataset_info'] + '/max_shape_' + dataset +'.pickle'
        pickle.dump(dictionary, open(pickle_filename, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
