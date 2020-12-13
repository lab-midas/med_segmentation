import tensorflow as tf
import numpy as np
import modAL
from scipy.stats import entropy

from pathlib import Path
from med_io.get_pad_and_patch import get_fixed_patches_index, pad_img_label, get_patches_data

"""
Active learning parts for pipeline
"""


def query_selection(model, X, n_classes):
    utility = segmentation_utility(model, X)
    pass


def segmentation_utility(model, X):
    pass


def predict_and_average_proba(model, X):
    predictions = model.predict(X)  # Problem - evtl zu viele Daten!!! sequence- enqueer?
    segmentation = np.mean(predictions, (1, 2, 3))
    return predictions


def utility_measure(prediction_proba, type):
    if type == 'entropy':
        return entropy(prediction_proba)
    elif type == 'uncertainty':
        return


""" functions that calculate an uncertainty measure from the class probability predictions 
    inspired by the corresponding functions in modAL uncertainty.py version 0.4.0 """


def _proba_uncertainty(proba):
    return 1 - np.max(proba, axis=-1)


def _proba_margin(proba):
    if proba.shape[-1] == 1:
        raise Exception('Not enought classes for margin uncertainty') # or set 0 like in modAL
    # sort the array in parts so that the first two elements in last axis are
    # the most certain predictions then return margin
    part = np.partition(-proba, 1, axis=-1)
    return - part[..., 0] + part[..., 1]


def _proba_entropy(proba):
    return entropy(proba, axis=-1)


""" ------------------------------- """


def query_training_patches(config, dataset_image_path, model, pool):
    """
    Patch the image data and calculate an active learning value
    (e.g. uncertainty of the network) for every patch, then select the n best
    patches with the highest value for training.
    """

    # predict data-patches
    #predict_patch_imgs = predict(config, model, patch_imgs, patches_indices)

    # calculate value of the patches for training
    #pool.calculate_values(predict_patch_imgs, patches_indices, image_number)

    # select the best n for training
    #pool.select_patches()

    return pool


def uncertainty_sampling(prediction, computation='entropy'):
    """
        Calculate an estimation of uncertainty for the prediction and return
        this value. The prediction data must be a Tensor with a probability
        value for every class for every voxel.
        :parm prediction: type tf.Tensor, 4D prediction data where the first 3
        Dimensions are Space and the 4th the class probabilities
        :return: uncertainty_value
    """
    # Calculate an uncertainty value for every pixel producing an uncertainty-field
    if computation == 'entropy':
        # calculate the Shannon Entropy for every pixel as uncertainty value
        probs_log = tf.math.log(prediction)
        weighted_probs_log = tf.math.multiply(prediction, probs_log)
        uncertainty_field = tf.math.reduce_sum(tf.math.negative(weighted_probs_log), axis=-1)
    elif computation == 'least_confident':
        # pick the probability of the most likely class as uncertainty value
        uncertainty_field = tf.reduce_max(prediction, axis=-1)
    else:
        raise Exception('Unknown way of computing the uncertainty')

    # Average the values to get an average uncertainty for the entire prediction
    uncertainty = tf.math.reduce_mean(uncertainty_field)
    return uncertainty


# Define classes to manage patch selection
class PatchPool:
    def __init__(self, config, dataset, dataset_image_path, batch_size=20):
        self.batch_size = batch_size
        self.dataset = dataset
        # mode for building pipeline with unused patches form pool or patches
        # to train: either 0 for query or 1 for train mode
        self.mode = 0

        # determine general patch indices and parameters for patching
        assert not config['patch_probability_distribution']['use']  #!evtl möglich? aber nicht sinnvoll?! make sure tiling method is used, if random shift should get turned on at some point make shure get_pos_key and communicatin of patches to pipeline still works
        self.patch_size = config['patch_size']
        self.dim = len(self.patch_size)
        self.max_data_size = [config['max_shape']['image'][i] for i in range(self.dim)]
        # general patch locations for max size image
        self.ideal_patches_indices = get_fixed_patches_index(config, self.max_data_size, self.patch_size,
                                                             overlap_rate=config['patch_overlap_rate'],
                                                             start=config['patch_start'],
                                                             end=config['patch_end'])
        # check what input channels are used (for creation of patches)
        if config['input_channel'][self.dataset] is not None:
            self.input_slice = config['input_channel'][self.dataset]

        # create the lists that keep track of the patches and their status for every image
        # status is one of 0 (unused), 1 (to train), 2 (used) and initialized as 0
        self.pool = {}
        for [image_path] in dataset_image_path:
            image_pathlib_path = Path(image_path)
            image_number = image_pathlib_path.parts[-3]
            # create list with patches indices in first 3 column an status in 4th
            self.pool[image_number] = np.zeros((len(self.ideal_patches_indices), 4), dtype=int)
            self.pool[image_number][:, :3] = self.ideal_patches_indices

    def get_patches_pipeline(self, image_data_path):
        image_pathlib_path = Path(image_data_path)
        image_number = image_pathlib_path.parts[-3]
        patch_list = self.pool[image_number]
        patch_list_for_pipeline = []
        patch_id = []
        for i, patch in enumerate(patch_list):
            # if index has the right status add the index to the returned list
            if patch[3] == self.mode:
                patch_list_for_pipeline.append(patch[:3])
                # add the parameters needed to identify the patch
                patch_id.append((str(i), image_number))
        return patch_list_for_pipeline, patch_id
