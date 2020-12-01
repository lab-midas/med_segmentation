import tensorflow as tf
import numpy as np
from math import ceil
from pathlib import Path
from med_io.parser_tfrec import parser
from med_io.get_pad_and_patch import get_fixed_patches_index, pad_img_label, get_patches_data

"""
Active learning parts for pipeline
"""


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


def predict(config, model, patch_imgs, patches_indices, img_data_shape=None):
    # reformat list of indices (inspired by predict.py line 184)
    indice_list_model = np.float32(patches_indices)
    if config['regularize_indice_list']['max_shape']:
        indice_list_model = np.float32(np.array(patches_indices)) / np.array(
            config['max_shape']['image'])[..., 0]
    elif config['regularize_indice_list']['image_shape']:
        indice_list_model = np.float32(np.array(patches_indices)) / np.array(
            img_data_shape)[:-1]  # !not yet fully implemented
    elif config['regularize_indice_list']['custom_specified']:
        indice_list_model = np.float32(np.array(patches_indices)) / np.array(
            config['regularize_indice_list']['custom_specified'])

    # predict data
    predict_patch_imgs = model.predict(x=(patch_imgs, indice_list_model),
                                       batch_size=1, verbose=1)
    return predict_patch_imgs


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
        # mode for building pipeline with
        # unused patches form pool or patches to train: either 'query' or 'train'
        self.mode = 'query'

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

    # def get_unused_patches_indices(self, image_number):
    #     if self.patches_set_up[image_number]:
    #         patches = self.pool[image_number]
    #         patches = list(patches.values())
    #         patches = list(map(lambda x: x.index, patches))
    #     else:
    #         patches = self.ideal_patches_indices
    #     return patches
    #
    # def select_patches(self):
    #     patches = []
    #     for image_key in self.pool:
    #         patches.extend(list(self.pool[image_key].values()))
    #     for n in range(self.batch_size):
    #         most_uncertain = max(patches, key=lambda x: x.uncertainty)
    #         patches.remove(most_uncertain)
    #         self.to_train[most_uncertain.image_number].append(most_uncertain)
    #         self.pool[most_uncertain.image_number].pop(self.get_pos_key(most_uncertain.index))
    #
    # def calculate_values(self, predict_patch_imgs, patches_indices, image_number):
    #     if not self.patches_set_up[image_number]:
    #         for index in patches_indices:
    #             self.pool[image_number][self.get_pos_key(index)] = Patch(image_number, index)
    #         self.patches_set_up[image_number] = True
    #
    #     for predict_patch, patch_index in zip(predict_patch_imgs, patches_indices):
    #         self.pool[image_number][self.get_pos_key(patch_index)].uncertainty = \
    #             uncertainty_sampling(predict_patch)
    #
    # def get_patches_to_train(self, image_number):
    #     patches = self.to_train.pop(image_number)
    #     self.to_train[image_number] = []
    #     self.used.append(patches)
    #     patches = list(map(lambda x: x.index, patches))
    #     return patches

    # maybe as dict key, less space
    def get_pos_key(self, index):
        key = 0
        for i in range(len(index)):
            key += int(ceil(index[i] / self.patch_size[i]) * (10 ** (2 * i)))
            # assumes less than 100 patches in every dimension!
        return key


class ImagePatches:
    def __init__(self, number, ideal_patches_indices, patch_pool):
        self.patch_pool = patch_pool
        self.number = number
        self.patches_indices = ideal_patches_indices
        self.patches_set_up = False
        self.pool = {}
        self.to_train = []

    def set_up_patches(self, patches_indices):
        self.patches_indices = patches_indices
        for index in self.patches_indices:
            self.pool[self.patch_pool.get_pos_key(index)] = Patch(self.number, index)
        self.patches_set_up = True

    def get_pipeline_patches_indices(self):
        if self.patch_pool.mode == 'query':
            if self.patches_set_up:
                patches = list(self.pool.values())
                patches = list(map(lambda x: x.index, patches))
            else:
                patches = self.patches_indices
            return patches
        elif self.patch_pool.mode == 'train':
            patches = list(map(lambda x: x.index, self.to_train))
            self.to_train = []
            return patches
        else:
            raise Exception('Unknown mode of PatchPool')

class Patch:
    """ Object that uniquely identifies a patch"""

    def __init__(self, image_number, index):
        self.image_number = image_number
        self.index = index
        self.uncertainty = 0

    def set_uncertainty(self, uncertainty):
        self.uncertainty = uncertainty
