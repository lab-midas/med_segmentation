import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K
import os

def read_info(index):
    path = 'test_img/elem_' + str(index) + '.pickle'
    #path = 'tests/tests/test_img/' + str(index) + '_elem.pickle'
    with open(path, 'rb') as handle:
        b = pickle.load(handle)
    return b['image'], b['mask']

def read_info_training(index):
    path = 'test_img/elem_train_' + str(index) + '.pickle'
    #path = 'tests/tests/test_img/' + str(index) + '_elem.pickle'
    with open(path, 'rb') as handle:
        b = pickle.load(handle)
    return b['image'], b['mask']

def read_info_validation(index):
    path = 'test_img/elem_validation_' + str(index) + '.pickle'
    #path = 'tests/tests/test_img/' + str(index) + '_elem.pickle'
    with open(path, 'rb') as handle:
        b = pickle.load(handle)
    return b['image'], b['mask']

def get_indexes(path_imgs):
    files = os.listdir(path_imgs)
    indexes = [file.split('_')[1].split('.')[0] for file in files]
    return indexes

def get_slices(img, slice_cut='s', patch=96):
    slices = []
    patches = np.linspace(0, patch - 1, int(patch / 10))
    if slice_cut == 's':
        for index in patches:
            slices.append(img[:, :, int(index)])

    if slice_cut == 'a':
        for index in patches:
            slices.append(img[:, int(index), :])

    if slice_cut == 'c':
        for index in patches:
            slices.append(img[int(index), :, :])

    return slices

def plot_img_mask(img, mask, slice_cut, patch, index=None):
    img_slices = get_slices(img, slice_cut, patch=patch)
    mask_slices = get_slices(mask, slice_cut, patch=patch)
    for slice_img, slice_mask in zip(img_slices, mask_slices):
        fig, ax = plt.subplots(nrows=2, ncols=2)
        fig.suptitle("elem: "+str(index))
        ax[0, 0].imshow(slice_img)
        ax[0, 0].set_title(slice_cut + " image")
        ax[0, 1].imshow(slice_mask)
        ax[0, 1].set_title(slice_cut + " mask")
        ax[1, 0].imshow(slice_img, cmap='gray')
        ax[1, 0].set_title(slice_cut + " image")
        ax[1, 1].imshow(slice_mask, cmap='gray')
        ax[1, 1].set_title(slice_cut + " mask")
        plt.show()

def test_patch():
    path_imgs = 'test_img/'
    #index_patches = get_indexes(path_imgs)
    index_patches = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    slice_cut = ['a']
    patch_size = [96, 96, 96]
    for index in index_patches:
        for cut in slice_cut:
            #img, mask = read_info(index)
            img, mask = read_info_training(index)
            #img, mask = read_info_validation(index)
            print(img.shape)
            print(mask.shape)
            for p in range(img.shape[0]):
                print("values stored in mask: ", np.unique(mask[p, ..., 1].numpy()))
                plot_img_mask(img[p, ..., 1], mask[p, ..., 1], cut, patch=patch_size[0], index=index)

    print("done")

##-- in case the test is to be performed, uncomment the next line and run just the test file
#test_patch()