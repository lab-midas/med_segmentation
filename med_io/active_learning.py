import tensorflow as tf
import numpy as np
from modAL.utils.selection import multi_argmax
from scipy.stats import entropy
from med_io.keras_data_generator import DataGenerator

"""
Active learning parts for training
"""


def query_selection(model, X, config, n_instances=1):
    """
        Query the ids of the most promising data
        :parm model: segmentation model that is supposed to be trained by al loop
        :parm X: Data for prediction e.g. DataGenerator object
        :parm config: config parameters
        :return: indices of the queried (best) data
        Note: the ids returned are the indices in the data provided, not the
              indices of the hdf5 file used in CustomActiveLearner!
    """
    # choose the type of utility function used for calculation of utility
    utility_functions = {'entropy': _proba_entropy,
                         'uncertainty': _proba_uncertainty,
                         'margin': _proba_margin}
    utility_function = utility_functions[config['information_estimation']]

    # choose how segmentation is condensed to a single utility value (using utility function from above)
    reduction_functions = {'value_of_means': _value_of_means,
                           'mean_of_values': _mean_of_values}
    reduction_function = reduction_functions[config['reduce_segmentation']]

    # utility evaluation using the predictions of the model for the data
    predictions = model.predict(X, workers=5, use_multiprocessing=True)
    utilities = reduction_function(predictions, utility_function)

    # selecting the best instances
    query_idx = multi_argmax(utilities, n_instances=n_instances)

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
    :init_ids: ids of data with which the model gets trained before al starts
    """
    def __init__(self, config, model, query_strategy, hdf5_path, pool_ids,
                 val_dataset, dataset, fit_batch_size, predict_batch_size,
                 init_ids=None):
        self.model = model
        self.query_strategy = query_strategy
        self.hdf5_path = hdf5_path
        # create the list that 0
        self.pool_ids = pool_ids
        self.train_ids = []
        # for creating the DataGenerator objects
        self.n_channels = len(config['input_channel'][dataset])
        self.n_classes = len(config['output_channel'][dataset])
        self.fit_batch_size = fit_batch_size
        self.predict_batch_size = predict_batch_size
        # train on initial data if given
        if init_ids is not None:
            self._fit_on_new(init_ids)
            self.train_ids.append(init_ids)
        self.val_dataset = val_dataset
        self.histories = []

    def _fit_on_new(self, ids, **fit_kwargs):
        """
        Fit the model to the data with given ids (data is saved in hdf5 file),
        save history in history attribute
        """
        data_generator = DataGenerator(self.hdf5_path, ids,
                                       n_channels=self.n_channels,
                                       n_classes=self.n_classes,
                                       batch_size=self.fit_batch_size,
                                       shuffle=True)
        history = self.model.fit(x=data_generator, validation_data=self.val_dataset, **fit_kwargs)
        self.histories.append(history)

    def _add_training_data(self, ids, label_data=None):
        """
        Remove the ids of new training data from pool and add to train_ids
        (To be implemented: if new label data is given after query save it in hdf5 file first)
        """
        if label_data is not None:
            # later add option to add label data to hdf5 file for unlabled data
            pass
        self.train_ids.append(ids)
        for train_id in ids:
            self.pool_ids.remove(train_id)

    def query(self, *query_args, **query_kwargs):
        """
        Query the ids of most promising data with help of the query_strategy
        """
        pool_data = DataGenerator(self.hdf5_path, self.pool_ids,
                                  n_channels=self.n_channels,
                                  n_classes=self.n_classes,
                                  batch_size=self.predict_batch_size,
                                  shuffle=False)
        query_result = self.query_strategy(self.model, pool_data,
                                           *query_args, **query_kwargs)
        # indices returned by query strategy note position in pool_ids not ids themself!
        query_ids = [self.pool_ids[i] for i in query_result]
        return query_ids

    def teach(self, ids, label_data=None, **fit_kwargs):
        """
        Add the ids of new training data to list of training data ids (and if
        provided add new label data to data in hdf5 file), then fit the model to
        the new data
        """
        self._add_training_data(ids, label_data=label_data)
        self._fit_on_new(ids, **fit_kwargs)
