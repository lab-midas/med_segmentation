import tensorflow as tf
import numpy as np
import keras
import h5py
import pickle
from pathlib import Path
from med_io.pipeline import pipeline


def tf_records_as_hdf5(dataset_train_image_path, dataset_train_label_path,
                       dataset_val_image_path, dataset_val_label_path,
                       config, dataset=None):
    """
    Load and patch the tf record data (train and val) through the pipeline.py
    and save the patches in a hdf5 file. Filter out patches that only contain 0
    (i.e. no information). Number every patch with a unique ID. Create lists that
    include all IDs of train and val patches.
    Note: if the path to the hdf5 file (as defined in config) already exists the
    function assumes the data was already converted and uses the hdf5 file that
    is already there!
    :dataset_train_image_path: paths to tf record files of train images
    :dataset_train_label_path: paths to tf record files of train labels
    :dataset_val_image_path: paths to tf record files of val images
    :dataset_val_label_path: paths to tf record files of val labels
    :config: dict with config parameters
    :dataset: name of the dataset used
    :return: path to the hdf5 file, list of train IDs, list of val IDs
    """
    # create file path, where the data is (going to be) stored
    hdf5_path = Path(config['al_patches_data_dir'])
    hdf5_path.mkdir(exist_ok=True)
    hdf5_path = hdf5_path / config['al_patches_data_file']

    # check if file already exists / if data was already converted
    if hdf5_path.is_file():
        # get id lists from saved data
        print(' File {0} already exists, using data stored there'.format(hdf5_path))
        with h5py.File(hdf5_path, 'r') as f:
            train_ids = list(f['id_lists']['train_ids'])
            val_ids = list(f['id_lists']['val_ids'])
    else:
        # convert tf_record data to hdf5 and return the id lists
        print(' Patching and converting data to hdf5 for al process')
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
        patch_info = {'fields': ('image number', 'patch position',
                                 'image path', 'label path')}
        for patch_num, (image_data, label_data, patch_pos) in dataset_train.enumerate(0):
            image_data = image_data.numpy()
            label_data = label_data.numpy()
            grp_images.create_dataset(str(patch_num.numpy()), data=image_data)
            grp_labels.create_dataset(str(patch_num.numpy()), data=label_data)
            # save info about the origin of the patch
            img_num = patch_num//config['max_patch_num']
            patch_info[str(patch_num.numpy())] = (img_num, patch_pos,
                                                  dataset_train_image_path[img_num],
                                                  dataset_train_label_path[img_num])
            # filter out the patches that only contain padding/ zeros from id list
            if not contains_only_zeros(image_data):
                train_ids.append(str(patch_num.numpy()))
        # save the indices of all training data in a list (conversion to ascii necessary)
        grp_id_lists.create_dataset('train_ids', data=[s.encode('ascii') for s in train_ids])

        img_num_offset = img_num+1

        # validation data: get the data from pipeline and store as hdf5
        for patch_num, (image_data, label_data, patch_pos) in dataset_val.enumerate(patch_num + 1):
            image_data = image_data.numpy()
            label_data = label_data.numpy()
            grp_images.create_dataset(str(patch_num.numpy()), data=image_data)
            grp_labels.create_dataset(str(patch_num.numpy()), data=label_data)
            # save info about the origin of the patch
            img_num = patch_num//config['max_patch_num']
            patch_info[str(patch_num.numpy())] = (img_num, patch_pos,
                                                  dataset_val_image_path[img_num-img_num_offset],
                                                  dataset_val_label_path[img_num-img_num_offset])
            # filter out the patches that only contain padding/ zeros from id list
            if not contains_only_zeros(image_data):
                val_ids.append(str(patch_num.numpy()))
        # save the indices of all validation data in a list (conversion to ascii necessary)
        grp_id_lists.create_dataset('val_ids', data=[s.encode('ascii') for s in val_ids])

    # save the infos of the patches in a separate file (dict in hdf5 not possible)
    patches_info_path = hdf5_path.with_name(hdf5_path.stem + '-patches_info.pickle')
    with open(patches_info_path, 'wb') as f:
        pickle.dump(patch_info, f)

    return train_ids, val_ids


def contains_only_zeros(image_data):
    """
    Determines if data patch should be omitted from AL training. Patches in the
    border region of the image can contain only padding (only values of 0). This
    messes with the AL query strategy because the model is very uncertain, but
    the patches don't contribute to training because they contain no real data.
    :param image_data: numpy array with image data
    :return: boolean value, True if image_data contains only 0, False otherwise
    """
    non_zero_values = image_data[image_data != 0]
    return (non_zero_values.size == 0)


def save_used_patches_ids(config, name, info, first_time=False):
    """
    :param config: config parameters from config file
    :param name: name under which the data should be saved e.g. epoch
    :param info: data to be saved under name, i.e. IDs of the patches used
    :param first_time: type bool; True if this is the first time to save data in
                       this path the function will create the pickle file and
                       save the name of the hdf5 file where the data is stored
    Save the specified info in a dict and dump it in a pickle file i.e. the IDs
    to enable later analysis of the data used in the training.
    Note: info and name can also be passed a list (make sure they have the same
          length) to add multiple pairs of name:info
    """
    dir_path = Path(config['result_rootdir'], 'patches_info')
    dir_path.mkdir(parents=True, exist_ok=True)

    save_path = dir_path / (config['exp_name'] + '-patches_info.pickle')

    # get dict with info from pickle file (or create if first time)
    if first_time:
        patches_data = {}
    else:
        with open(save_path, 'rb') as f:
            patches_data = pickle.load(f)

    # save the information under the corresponding names
    if isinstance(name, list) and isinstance(info, list):
        for n, i in zip(name, info):
            patches_data[n] = i
    else:
        patches_data[name] = info

    # save the dict in pickle file again
    with open(save_path, 'wb') as f:
        pickle.dump(patches_data, f)


"""The following code is inspired by the blog post https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly# 
    and is an adapted version of code provided there"""


class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, hdf5_data_path, list_IDs, batch_size=32, dim=(32, 32, 32), n_channels=1,
                 n_classes=4, shuffle=True, steps_per_epoch=None):
        """
        Initialize DataGenerator object
        :hdf5_data_path: path where the data (patches) are stored
        :list_IDs: list with the IDs that identify the patches to be used in the
        hdf5 file
        :batch_size: batch size
        :dim: type tupel, dimensions of the patches
        :n_channels: number of channels in the input image
        :n_classes: number of classes in the output (label)
        :shuffle: if True, shuffle the patches after every epoch
        :steps_per_epoch: number of batches per epoch, use max number of batches
        if steps_per_epoch is None
        """
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.list_IDs = list_IDs
        self.hdf5_data_path = hdf5_data_path
        self.on_epoch_end()
        self.steps_per_epoch = steps_per_epoch

    def __len__(self):
        'return the number of batches per epoch'
        if self.steps_per_epoch is None:
            return int(np.floor(len(self.list_IDs) / self.batch_size))
        else:
            return self.steps_per_epoch

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
