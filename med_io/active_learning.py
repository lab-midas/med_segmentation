import tensorflow as tf
import numpy as np
from modAL.utils.selection import multi_argmax
from scipy.stats import entropy
from med_io.keras_data_generator import DataGenerator

"""
Active learning parts for pipeline
"""


def query_selection(model, X, config):
    # choose the type of utility function used for calculation of utility
    utility_functions = {'entropy': _proba_entropy,
                         'uncertainty': _proba_uncertainty,
                         'margin': _proba_margin}
    utility_function = utility_functions[config['information_estimation']]

    utilities = value_of_means(model, X, utility_function)

    # selecting the best instances
    query_idx = multi_argmax(utilities)

    return query_idx


def value_of_means(model, X, utility_function):
    predictions = model.predict(X)  # Problem - evtl zu viele Daten!!! sequence- enqueer?
    mean_predictions = np.mean(predictions, (1, 2, 3))
    utilities = utility_function(mean_predictions)
    return utilities


def mean_of_values(model, X, utility_function):
    predictions = model.predict(X)  # Problem - evtl zu viele Daten!!! sequence- enqueer?
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
    def __init__(self, config, model, query_strategy, hdf5_path, pool_ids,
                 val_dataset, dataset, init_ids=None):
        self.model = model
        self.query_strategy = query_strategy
        self.hdf5_path = hdf5_path
        # create the list that 0
        self.pool_ids = pool_ids
        self.train_ids = []
        # for creating the DataGenerator objects
        self.n_channels = len(config['input_channel'][dataset])
        self.n_classes = len(config['output_channel'][dataset])
        self.batch_size = 2
        # train on initial data if given
        if init_ids is not None:
            self._fit_on_new(init_ids)
            self.train_ids.append(init_ids)
        self.val_dataset = val_dataset
        self.histories = []

    def _fit_on_new(self, ids):
        data_generator = DataGenerator(self.hdf5_path, ids,
                                       n_channels=self.n_channels,
                                       n_classes=self.n_classes,
                                       batch_size=self.batch_size)
        history = self.model.fit(x=data_generator, validation_data=self.val_dataset)
        self.historys.append(history)

    def _add_training_data(self, ids, label_data=None):
        if label_data is not None:
            # later add option to add label data to hdf5 file for unlabled data
            pass
        self.train_ids.append(ids)
        for train_id in ids:
            self.pool_ids.remove(train_id)

    def query(self, *query_args, **query_kwargs):
        pool_data = DataGenerator(self.hdf5_path, self.pool_ids,
                                  n_channels=self.n_channels,
                                  n_classes=self.n_classes,
                                  batch_size=self.batch_size)
        query_result = self.query_strategy(self.model, pool_data, *query_args, **query_kwargs)
        return query_result

    def teach(self, ids, label_data=None):
        self._add_training_data(ids, label_data=label_data)
        self._fit_on_new(ids)
