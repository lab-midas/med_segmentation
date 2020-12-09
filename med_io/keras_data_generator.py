import tensorflow as tf
import numpy as np
import keras
import h5py
from pathlib import Path
from med_io.pipeline import pipeline


def tf_records_as_hdf5(dataset_train_image_path, dataset_train_label_path,
                       dataset_val_image_path, dataset_val_label_path,
                       config, dataset=None):
    # create file path, where the data is (going to be) stored
    hdf5_path = Path(config['al_patches_data_dir'])
    hdf5_path.mkdir(exist_ok=True)
    hdf5_path = hdf5_path / 'al_patches_data3.hdf5'

    # check if file already exists / if data was already converted
    if hdf5_path.is_file():
        # get id lists from saved data
        with h5py.File(hdf5_path, 'r') as f:
            train_ids = list(f['id_lists']['train_ids'])
            val_ids = list(f['id_lists']['val_ids'])
    else:
        # convert tf_record data to hdf5 and return the id lists
        train_ids, val_ids = convert_tf_records_hdf5(
            dataset_train_image_path, dataset_train_label_path,
            dataset_val_image_path, dataset_val_label_path,
            hdf5_path, config, dataset=dataset)
    return hdf5_path, train_ids, val_ids


def convert_tf_records_hdf5(dataset_train_image_path, dataset_train_label_path,
                            dataset_val_image_path, dataset_val_label_path,
                            hdf5_path, config, dataset=None):
    """ Read image and label data from TFRecords, patch the data and save it as
        hdf5 data. The process is carried out through the tf pipeline in pipeline.py"""
    # get a dataset with the patches from TFRecords through pipeline
    dataset_train = pipeline(config, dataset_train_image_path, dataset_train_label_path,
                             dataset=dataset, no_shuffle_and_batching=True)
    dataset_val = pipeline(config, dataset_val_image_path, dataset_val_label_path,
                           dataset=dataset, no_shuffle_and_batching=True)

    # save the patches data (of the dataset) as a hdf5 file in home dir
    with h5py.File(hdf5_path, 'w') as f:
        # create the hdf5 groups that store the datasets
        grp_images = f.create_group('images')
        grp_labels = f.create_group('labels')
        grp_id_lists = f.create_group('id_lists')

        # training data: get the data from pipeline and store as hdf5
        train_ids, val_ids = [], []
        for img_num, (image_data, label_data) in dataset_train.take(4).enumerate(0):  # faster solution?
            image_data = image_data.numpy()
            label_data = label_data.numpy()
            grp_images.create_dataset(str(img_num.numpy()), data=image_data)
            grp_labels.create_dataset(str(img_num.numpy()), data=label_data)
            train_ids.append(str(img_num.numpy()))
        # save the indices of all training data in a list (conversion to ascii necessary)
        grp_id_lists.create_dataset('train_ids', data=[s.encode('ascii') for s in train_ids])

        # validation data: get the data from pipeline and store as hdf5
        for img_num, (image_data, label_data) in dataset_val.take(4).enumerate(img_num + 1):
            image_data = image_data.numpy()
            label_data = label_data.numpy()
            grp_images.create_dataset(str(img_num.numpy()), data=image_data)
            grp_labels.create_dataset(str(img_num.numpy()), data=label_data)
            val_ids.append(str(img_num.numpy()))
        grp_id_lists.create_dataset('val_ids', data=[s.encode('ascii') for s in val_ids])

    return train_ids, val_ids


"""This code is inspired by the blog post https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly# 
    and is an adapted version of code provided there"""


class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, hdf5_data_path, list_IDs, batch_size=32, dim=(32, 32, 32), n_channels=1,
                 n_classes=4, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.list_IDs = list_IDs
        self.hdf5_data_path = hdf5_data_path
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim, self.n_classes))

        # Generate data
        with h5py.File(self.hdf5_data_path, 'r') as f:
            for i, ID in enumerate(list_IDs_temp):
                # Store sample
                X[i,] = f['images'][ID]

                # Store class
                y[i] = f['labels'][ID]

        return X, y
