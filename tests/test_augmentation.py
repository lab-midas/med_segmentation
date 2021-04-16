import numpy as np
import tensorflow as tf
from tests.test_patch import get_slices
from tests.test_loss_function import *
from med_io.augmentation import *
from med_io.augmentation_batch import *
import matplotlib.pyplot as plt


def plot_img_augmentation(img, slice_cut, patch):
    img_slices = get_slices(img, slice_cut, patch=patch)
    img_aug = tf.convert_to_tensor(np.expand_dims(img, axis=0))

    img_bright = brightness_transform(img_aug, mu=0.9, sigma=0.35, num_patches=1, num_channel=2,
                                      shape_data=img.shape, per_channel=True, p_per_channel=1)

    img_gamma = gamma_contrast(img_bright, [96, 96, 96], num_patches=1, num_channel=2, shape_data=None,
                               gamma_range=(1.6, 1.8), invert_image=False, per_channel=False,
                               retain_stats=False)

    img_contrast = contrast_augmentation_transform(img_bright, contrast_range=(1.5, 1.7), num_patches=1, num_channel=2,
                                                   shape_data=img.shape, preserve_range=True, per_channel=True,
                                                   p_per_sample=1)
    img_gamma_slices = get_slices(img_gamma[0, ...], slice_cut, patch=patch)
    img_bright_slices = get_slices(img_bright[0, ...], slice_cut, patch=patch)
    img_contrast_slices = get_slices(img_contrast[0, ...], slice_cut, patch=patch)
    for slice_img, slice_gamma, slice_brigh, slice_cont in zip(img_slices, img_gamma_slices, img_bright_slices, img_contrast_slices):
        fig, ax = plt.subplots(nrows=2, ncols=2)
        ax[0, 0].imshow(slice_img[..., 0])
        ax[0, 0].set_title(slice_cut + " normal")
        ax[0, 1].imshow(slice_gamma[..., 0])
        ax[0, 1].set_title(slice_cut + " brightness augmentation")
        ax[1, 0].imshow(slice_gamma[..., 0])
        ax[1, 0].set_title(slice_cut + " gamma augmentation")
        ax[1, 1].imshow(slice_cont[..., 0])
        ax[1, 1].set_title(slice_cut + " contrast augmentation")
        plt.show()

def plot_img_augmentation_batch(img, slice_cut, patch):
    img_slices = get_slices(img, slice_cut, patch=patch)
    img_aug = np.rollaxis(img, axis=3, start=0)
    img_gamma = augment_gamma(img_aug, gamma_range=(1.6, 1.8), invert_image=False, per_channel=False,
                              retain_stats=False)
    img_bright = augment_brightness_additive(img_gamma, mu=0.9, sigma=0.3, per_channel=True, p_per_channel=1) #.numpy()
    img_contrast = augment_contrast(img_bright, contrast_range=(1.5, 1.7), preserve_range=False, per_channel=True) #.numpy()

    # time to make the slices
    img_gamma_new = np.rollaxis(img_gamma, axis=0, start=4)
    img_brigh_new = np.rollaxis(img_bright, axis=0, start=4)
    img_cont_new = np.rollaxis(img_contrast, axis=0, start=4)

    img_gamma_slices = get_slices(img_gamma_new, slice_cut, patch=patch)
    img_bright_slices = get_slices(img_brigh_new, slice_cut, patch=patch)
    img_contrast_slices = get_slices(img_cont_new, slice_cut, patch=patch)

    for slice_img, slice_gamma, slice_brigh, slice_cont in zip(img_slices, img_gamma_slices, img_bright_slices, img_contrast_slices):
        fig, ax = plt.subplots(nrows=2, ncols=2)
        ax[0, 0].imshow(slice_img[..., 0])
        ax[0, 0].set_title(slice_cut + " normal batch")
        ax[0, 1].imshow(slice_gamma[..., 0])
        ax[0, 1].set_title(slice_cut + " gamma augmentation batch")
        ax[1, 0].imshow(slice_brigh[..., 0])
        ax[1, 0].set_title(slice_cut + " brightness augmentation batch")
        ax[1, 1].imshow(slice_cont[..., 0])
        ax[1, 1].set_title(slice_cut + " contrast augmentation batch")
        plt.show()

def test_augmentation():
    path_imgs = 'tests/test_img/'
    #path_imgs = 'tests/tests/test_img/'
    indexes = [get_indexes(path_imgs)[2]]
    slice_cut = 'a'
    patch_size = 96

    for index in indexes:
        patch_img, label = read_info(index)
        plot_img_augmentation(patch_img, slice_cut, patch=patch_size)
        plot_img_augmentation_batch(patch_img, slice_cut, patch=patch_size)


##-- in case the test is to be performed, uncomment the next line and run just the test file
#test_augmentation()
