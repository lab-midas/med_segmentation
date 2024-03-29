import tensorflow as tf
import numpy as np
import pickle
import random
import os
from pathlib import Path
from modAL.utils.selection import multi_argmax
from scipy.stats import entropy
from med_io.keras_data_generator import DataGenerator, save_used_patches_ids

"""
Active learning parts for training
"""


def choose_random_elements(_list, num_elements=5000):
    """
    Choose a certain number of elements from given list randomly and return the
    original list without the chosen elements as well as a list of the chosen
    elements.
    :param _list: list with elements to be chosen from
    :param num_elements: number of elements to be chosen
    :return: tuple (original list without elements, list with chosen elements)
    """
    indices = random.sample(range(len(_list)), k=num_elements)
    choices = []
    for index in sorted(indices, reverse=True):
        choices.append(_list.pop(index))
    return _list, choices


def query_random(*args, n_instances=1, nr_patches_in_pool=None, **kwargs):
    """
    Select random ids from supplied data
    :param nr_patches_in_pool: Nr of patches that are in the pool
    :param n_instances: number of instances to be queried
    :return: List of randomly selected ids of instances from pool
    Note: the arguments X, model and config are only for consistency and are not
          used in the function
    """
    return random.sample(range(nr_patches_in_pool), k=n_instances)


def query_selection(model, X, config, n_instances=1, al_epoch=None,
                    al_num_workers=5, **kwargs):
    """
        Query the ids of the most promising data
        :parm model: segmentation model that is supposed to be trained by al loop
        :parm X: Data for prediction, type list of Datasets e.g. DataGenerator objects
                The form as list is so that never too much data at once is in calculation
        :parm config: config parameters
        :return: indices of the queried (best) data
        Note: the ids returned are the indices of the position in the data
              provided, not the indices of the hdf5 file used in
              CustomActiveLearner!
    """
    # choose the type of utility function used for calculation of utility
    utility_functions = {'entropy': _proba_entropy,
                         'uncertainty': _proba_uncertainty,
                         'margin': _proba_margin}
    utility_function = utility_functions[config['information_estimation']]
    # choose how segmentation is condensed to a single utility value
    # (using utility function from above)
    reduction_functions = {'value_of_means': _value_of_means,
                           'mean_of_values': _mean_of_values}
    reduction_function = reduction_functions[config['reduce_segmentation']]

    # utility evaluation using the predictions of the model for the data
    utilities = np.array([])
    for data in X:
        print('Calculating utilities of {0} batches'.format(len(data)))
        predictions = model.predict(data, workers=al_num_workers,
                                    use_multiprocessing=True)
        _utilities = reduction_function(predictions, utility_function)
        utilities = np.concatenate((utilities, _utilities))

    # selecting the best instances
    query_idx = multi_argmax(utilities, n_instances=n_instances)

    # save utility values of queried instances
    pickle_path = Path(config['result_rootdir'],
                       'al_utilities' + '_' + config['exp_name'] + '.pickle')
    if not os.path.exists(pickle_path):
        with open(pickle_path, 'w'): pass
    with open(pickle_path, 'rb+') as f:
        if al_epoch == 0:
            data = np.empty((config['al_iterations'], n_instances))
            data[al_epoch] = utilities[query_idx]
        else:
            data = pickle.load(f)
            data[al_epoch] = utilities[query_idx]
        pickle.dump(data, f)

    return query_idx


""" functions that enable the reduction of an entire segmentation prediction to
    a single value, using one of the functions below (71-85)"""


def _value_of_means(predictions, utility_function):
    mean_predictions = np.mean(predictions, (1, 2, 3))
    utilities = utility_function(mean_predictions)
    return utilities


def _mean_of_values(predictions, utility_function):
    utilities = utility_function(predictions)
    mean_utilities = np.mean(utilities, (1, 2, 3))
    return mean_utilities


""" functions that calculate an utility value from the class probability predictions 
    inspired by the corresponding functions in modAL uncertainty.py version 0.4.0 """


def _proba_uncertainty(proba):
    return 1 - np.max(proba, axis=-1)


def _proba_margin(proba):
    if proba.shape[-1] == 1:
        raise Exception('Not enought classes for margin uncertainty')  # or set 0 like in modAL
    # sort the array in parts so that the first two elements in last axis are
    # the most certain predictions then return margin
    part = np.partition(-proba, 1, axis=-1)
    return - part[..., 0] + part[..., 1]


def _proba_entropy(proba):
    return entropy(proba, axis=-1)


# Define ActiveLearner class to manage active learning loop. The class is inspired
# by ActiveLearner class from modAL but designed to work with the DataGenerator class
class CustomActiveLearner:
    """
    Object that manages active learning
    :param config: config parameters
    :param model: keras model that is supposed to be trained
    :query_strategy: function that takes model and prediction data as input and
                     returns the indices of data to be queried
    :hdf5_path: path where the data and labels are saved
    :pool_ids: ids of data (in hdf5 file) that are available to be queried
    :val_dataset: tf dataset with validation data (doesn't work with Sequence object)
    :dataset: name of dataset used
    :train_steps_per_epoch: number of batches per epoch of training, use all if None
    :init_ids: ids of data with which the model gets trained before al starts
    :max_predict_num: max num of patches that are processed at once in query
                      (limited to avoid too high memory usage)
    """

    def __init__(self, config, model, query_strategy, hdf5_path, pool_ids,
                 dataset, fit_batch_size, predict_batch_size,
                 train_steps_per_epoch=None, init_ids=None,
                 max_predict_num=10000, **fit_kwargs):
        self.model = model
        self.query_strategy = query_strategy
        self.hdf5_path = hdf5_path
        # create the list that monitors patches data + param for saving used IDs
        self.pool_ids = pool_ids
        self.train_ids = []
        self.save_id_config = {k: config[k] for k in ['result_rootdir', 'exp_name']}
        # for creating the DataGenerator objects
        self.n_channels = len(config['input_channel'][dataset])
        self.n_classes = len(config['output_channel'][dataset])
        self.patch_size = config['patch_size']
        self.fit_batch_size = fit_batch_size
        self.predict_batch_size = predict_batch_size
        self.histories = []
        self.max_predict_num = max_predict_num
        self.train_steps_per_epoch = train_steps_per_epoch
        # keep track of epoch parameters
        self.fit_epoch_kwargs = {'epochs': config['epochs'],
                                 'initial_epoch': 0}
        self.epochs = config['epochs']
        # train on initial data if given
        if init_ids is not None:
            print('Training on init data')
            # fit_kwargs['callbacks'] = al_callbacks(config, 'init')
            self._fit_on_new(init_ids, **fit_kwargs)
            self.train_ids += init_ids

    def _fit_on_new(self, ids, **fit_kwargs):
        """
        Fit the model to the data with given ids (data is saved in hdf5 file),
        save history in history attribute
        """
        data_generator = DataGenerator(self.hdf5_path, ids,
                                       dim=self.patch_size,
                                       n_channels=self.n_channels,
                                       n_classes=self.n_classes,
                                       batch_size=self.fit_batch_size,
                                       shuffle=True,
                                       steps_per_epoch=self.train_steps_per_epoch)
        # save the ids of the patches used
        save_used_patches_ids(self.save_id_config,
                              'epoch' + str(self.fit_epoch_kwargs['initial_epoch']), ids)
        # fit on the data
        print('Training on new data, {0} patches'.format(len(ids)))
        history = self.model.fit(x=data_generator,
                                 **fit_kwargs,
                                 **self.fit_epoch_kwargs)
        # update epoch arguments
        self.fit_epoch_kwargs['epochs'] += self.epochs
        self.fit_epoch_kwargs['initial_epoch'] += self.epochs

        self.histories.append(history)

    def _fit_on_all(self, **fit_kwargs):
        """
        Fit the model to all data in the labeled set
        (data is saved in hdf5 file), save history in history attribute
        """
        data_generator = DataGenerator(self.hdf5_path,
                                       self.train_ids,
                                       dim=self.patch_size,
                                       n_channels=self.n_channels,
                                       n_classes=self.n_classes,
                                       batch_size=self.fit_batch_size,
                                       shuffle=True,
                                       steps_per_epoch=self.train_steps_per_epoch)
        # save the ids of the patches used
        save_used_patches_ids(self.save_id_config,
                              'epoch' + str(self.fit_epoch_kwargs['epochs']), self.train_ids)
        # fit on the data
        print('Training on all labeled data, {0} patches'.format(len(self.train_ids)))
        history = self.model.fit(x=data_generator,
                                 **fit_kwargs,
                                 **self.fit_epoch_kwargs)
        # update epoch arguments
        self.fit_epoch_kwargs['epochs'] += self.epochs
        self.fit_epoch_kwargs['initial_epoch'] += self.epochs

        self.histories.append(history)

    def _add_training_data(self, ids, label_data=None):
        """
        Remove the ids of new training data from pool and add to train_ids
        (To be implemented: if new label data is given after query save it in hdf5 file first)
        """
        if label_data is not None:
            # later add option to add label data to hdf5 file for unlabled data
            pass
        self.train_ids += ids
        for train_id in ids:
            self.pool_ids.remove(train_id)

        print('Added new patches; unlabeled pool: {0} ; labeled data: {1}'.format(
            len(self.pool_ids), len(self.train_ids)))

    def _get_split_pool(self):
        """
        Create the data generator objects of the data in the pool for querying,
        split the data in manageable parts if the number of patches is too big
        (according to max_predict_num)
        """
        # assure that length of split parts is multiple of batch size
        split_length = (self.max_predict_num // self.predict_batch_size) * self.predict_batch_size

        # split pool_ids list into manageable pieces
        num_of_pieces = len(self.pool_ids) // split_length
        split_pool_ids = [self.pool_ids[i * split_length:(i + 1) * split_length]
                          for i in range(num_of_pieces)]
        split_pool_ids.append(self.pool_ids[num_of_pieces * split_length:])

        # create a DataGenerator object for every split part
        pool_data_list = []
        for pool_ids in split_pool_ids:
            pool_data = DataGenerator(self.hdf5_path, pool_ids,
                                      dim=self.patch_size,
                                      n_channels=self.n_channels,
                                      n_classes=self.n_classes,
                                      batch_size=self.predict_batch_size,
                                      shuffle=False)
            pool_data_list.append(pool_data)
        return pool_data_list

    def query(self, **query_kwargs):
        """
        Query the ids of most promising data with help of the query_strategy
        """
        # build DataGenerator for prediction of data (split into manageable chunks)
        pool_data_list = self._get_split_pool()

        print('Querying new patches')
        query_result = self.query_strategy(self.model, pool_data_list,
                                           nr_patches_in_pool=len(self.pool_ids),
                                           **query_kwargs)
        # indices returned by query strategy note position in pool_ids not ids themself!
        query_ids = [self.pool_ids[i] for i in query_result]
        return query_ids

    def teach(self, ids, only_new=True, label_data=None, **fit_kwargs):
        """
        Add the ids of new training data to list of training data ids (and if
        provided add new label data to data in hdf5 file), then fit the model to
        the new data
        """
        print('teach new patches')
        self._add_training_data(ids, label_data=label_data)

        # train the model either only on the new data or on entire labeled set
        if only_new:
            self._fit_on_new(ids, **fit_kwargs)
        else:
            self._fit_on_all(**fit_kwargs)
