import tensorflow as tf
import numpy as np
import keras
import h5py
from pathlib import Path
from med_io.pipeline import pipeline


def convert_tf_records_hdf5(dataset_train_image_path, dataset_train_label_path,
                            dataset_val_image_path, dataset_val_label_path,
                            config, dataset=None):
    """ Read image and label data from TFRecords, patch the data and save it as
        hdf5 data. The process is carried out through the tf pipeline in pipeline.py"""
    # data_path_image_list = [t[i] for t in dataset_image_path for i in range(len(dataset_image_path[0]))]
    # data_path_label_list = [t[i] for t in dataset_label_path for i in range(len(dataset_label_path[0]))]
    # list_image_TFRecordDataset = [tf.data.TFRecordDataset(i) for i in data_path_image_list]
    # list_label_TFRecordDataset = [tf.data.TFRecordDataset(i) for i in data_path_label_list]

    # zip_data_path_TFRecordDataset = tf.data.Dataset.zip(
    #    (list_image_TFRecordDataset[0], list_label_TFRecordDataset[0]))

    # get a dataset with the patches from TFRecords through pipeline
    dataset_train = pipeline(config, dataset_train_image_path, dataset_train_label_path,
                             dataset=dataset, no_shuffle_and_batching=True)
    dataset_val = pipeline(config, dataset_val_image_path, dataset_val_label_path,
                           dataset=dataset, no_shuffle_and_batching=True)

    # save the patches data (of the dataset) as a hdf5 file in home dir
    hdf5_path = Path(config['al_patches_data_dir'])
    hdf5_path.mkdir(exist_ok=True)
    hdf5_path = hdf5_path / 'al_patches_data3.hdf5'
    with h5py.File(hdf5_path, 'w') as f:
        # create the hdf5 groups that store the datasets
        grp_images_train = f.create_group('images/train')
        grp_images_val = f.create_group('images/validation')
        grp_labels = f.create_group('labels')
        # training data: get the data from pipeline and store as hdf5
        for img_num, (image_data, label_data) in dataset_train.take(4).enumerate(0): # faster solution?
            image_data = image_data.numpy()
            label_data = label_data.numpy()
            grp_images_train.create_dataset(str(img_num.numpy()), data=image_data)
            grp_labels.create_dataset(str(img_num.numpy()), data=label_data)
        # validation data: get the data from pipeline and store as hdf5
        for img_num, (image_data, label_data) in dataset_val.take(4).enumerate(img_num + 1):
            image_data = image_data.numpy()
            label_data = label_data.numpy()
            grp_images_val.create_dataset(str(img_num.numpy()), data=image_data)
            grp_labels.create_dataset(str(img_num.numpy()), data=label_data)
    return hdf5_path
