import tensorflow as tf
import numpy as np
from math import ceil
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
    # Data loading inspired by predict.py

    # Prepare to read data from TFRecord Files
    # Reformat data path list: [[path1],[path2], ...] ->[[path1, path2, ...]]
    data_path_image_list = [t[i] for t in dataset_image_path for i in range(len(dataset_image_path[0]))]
    list_image_TFRecordDataset = [tf.data.TFRecordDataset(i) for i in data_path_image_list]

    # for every image, get patches and determine each ones uncertainty value
    for image_number, image_TFRecordDataset in enumerate(list_image_TFRecordDataset):
        #       img_data, img_shape = image_TFRecordDataset.map(parser)
        #       img_data = img_data.numpy()
        dataset_image = image_TFRecordDataset.map(parser)
        img_data = [elem[0].numpy() for elem in dataset_image][0]
        img_shape = [elem[1].numpy() for elem in dataset_image][0]
        img_data = pad_img_label(config, pool.max_data_size, img_data, img_shape)

        patch_imgs, _, patches_indices = get_patches_data(pool.max_data_size, pool.patch_size, img_data,
                                                          pool.get_unused_patches_indices(image_number),
                                                          slice_channel_img=pool.input_slice,
                                                          output_patch_size=config['model_output_size'],
                                                          random_shift_patch=False)

        # predict data-patches
        predict_patch_imgs = predict(config, model, patch_imgs, patches_indices)

        # calculate value of the patches for training
        pool.calculate_values(predict_patch_imgs, patches_indices, image_number)

        # only for debugging
        print(image_number)
        if image_number >= 0:
            break

    # select the best n for training
    pool.select_patches()

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
    def __init__(self, config, dataset, num_of_imgs, batch_size=20):
        self.batch_size = batch_size
        self.dataset = dataset

        # determine general patch indices and parameters for patching
        assert not config['patch_probability_distribution']['use']  # make sure tiling method is used, if random shift should get turned on at some point make shure get_pos_key and communicatin of patches to pipeline still works
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

        # create pools of patches that keep track which patches have been trained
        self.to_train = {}
        self.used = []
        self.pool = {}
        # in the beginning all patches for every image are considered
        for i in range(num_of_imgs):
            self.pool[i] = {}
            self.to_train[i] = []
        self.patches_set_up = [False] * num_of_imgs

    def get_unused_patches_indices(self, image_number):
        if self.patches_set_up[image_number]:
            patches = self.pool[image_number]
            patches = list(patches.values())
            patches = list(map(lambda x: x.index, patches))
        else:
            patches = self.ideal_patches_indices
        return patches

    def select_patches(self):
        patches = []
        for image_key in self.pool:
            patches.extend(list(self.pool[image_key].values()))
        for n in range(self.batch_size):
            most_uncertain = max(patches, key=lambda x: x.uncertainty)
            patches.remove(most_uncertain)
            self.to_train[most_uncertain.image].append(most_uncertain)
            self.pool[most_uncertain.image].pop(self.get_pos_key(most_uncertain.index))

    def calculate_values(self, predict_patch_imgs, patches_indices, image_number):
        if not self.patches_set_up[image_number]:
            for index in patches_indices:
                self.pool[image_number][self.get_pos_key(index)] = Patch(image_number, index)
            self.patches_set_up[image_number] = True

        for predict_patch, patch_index in zip(predict_patch_imgs, patches_indices):
            self.pool[image_number][self.get_pos_key(patch_index)].uncertainty = \
                uncertainty_sampling(predict_patch)

    def get_patches_to_train(self, image_number):
        patches = self.to_train.pop(image_number)
        self.to_train[image_number] = []
        self.used.append(patches)
        patches = list(map(lambda x: x.index, patches))
        return patches

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