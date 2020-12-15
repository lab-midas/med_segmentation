import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K
from med_io.read_HD5F import *

def normalize(img, globalscale=False, channel_at_beginning=False):
    """
    Change pixel values in img to (0,1)
    :param img: type ndarray: input images
    :param globalscale: type boolean: (True) perform normalization on whole 2D/3D image, (False) axis independent normalization
    :param channel_at_beginning: type boolean: channels are at the beginning of input shape
    :return:img: type ndarray
    """

    print("Shape to normalize is: ", img.shape)

    if not channel_at_beginning:

        if globalscale:
            maxval = np.amax(img)
            minval = np.amin(img)
            img = (img - minval) / (maxval - minval + 1E-16)


        else:
            img = [(img[..., i] - np.min(img[..., i])) / (np.ptp(img[..., i]) + 1E-16) for i in
                   range(img.shape[-1])]
        img = np.rollaxis(np.float32(np.array(img)), 0, 4)

    else:  # this means that the channel is on the first position (channels, x, y, z)

        num_channels = img.shape[3]
        # img = np.rollaxis(np.float32(np.array(img)), 0, 4) # changes the channel axis to the end
        print("Shape before normalization: ", img.shape)

        if globalscale:
            maxval = np.amax(img)
            minval = np.amin(img)
            img = (img - minval) / (maxval - minval + 1E-16)


        else:
            img = [(img[..., i] - np.min(img[..., i])) / (np.ptp(img[..., i]) + 1E-16) for i in
                   range(img.shape[-1])]

        img = np.rollaxis(np.float32(np.array(img)), 0, 4)
        print("Shape after normalization: ", img.shape)

        assert num_channels == img.shape[-1], "normalization was not good performed"

    print("Final Shape: ", img.shape)

    return img

def read_file():
    rootdir_file = '/mnt/data/rawdata/Melanom/Tumorvolume'
    Data_Reader = HD5F_Reader(dataset='MELANOM', rootdir_file=rootdir_file)

    img_IDs = Data_Reader.img_IDs
    file = Data_Reader.file
    return file, img_IDs

def read_get_image(file, img_IDs, index=0):

    img = file['image']

    img_h5 = file['image'][img_IDs[index]]
    print("Shape of the image is: ", img_h5.shape)
    print("Type of the image is: ", type(img_h5))
    # the form of the images are  (channel, H, W, D)

    img_array = np.rollaxis(np.float32(np.array(img_h5)), 0, 4)
    im_norm = normalize(img_array)
    print("Shape of the image ARRAY is: ", im_norm.shape)
    print("Type of the image rolled is: ", type(im_norm))
    im_norm = normalize(img_array)

    mask_h5 = file['mask'][img_IDs[index]]  # mask or label
    print("Shape of the mask_h5 is: ", mask_h5.shape)
    print("Type of the mask is: ", type(mask_h5))

    mask_array = np.rollaxis(np.float32(np.array(mask_h5)), 0, 4)
    print("Shape of the mask ARRAY is: ", mask_array.shape)
    print("Type of the mask rolled is: ", type(mask_array))

    mask_one_hot = tf.one_hot(tf.argmax(mask_array, axis=-1), 2)

    assert (img_array.shape == mask_one_hot.shape)

    return img_array, mask_one_hot


def get_slices(img, slice_cut='s'):
    slices = []
    if slice_cut == 's':
        for index in range(img.shape[-2]):
            slices.append(img[:, :, index, 0])

    if slice_cut == 'a':
        for index in range(img.shape[1]):
            slices.append(img[:, index, :, 0])

    if slice_cut == 'c':
        for index in range(img.shape[0]):
            slices.append(img[index, :, :, 0])

    return slices


def plot_img(img, slice, slice_cut, name):
    img_slices = get_slices(img, slice_cut)
    slice = slice
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].imshow(img_slices[slice])
    ax[0].set_title(slice_cut + " normal, slice: " + str(slice))
    ax[1].imshow(img_slices[slice])
    ax[1].set_title(name)
    plt.show()


def plot_img_after_augmentation(img, slice, slice_cut, name):
    img_slices = get_slices(img, slice_cut)
    slice = slice
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.imshow(img_slices[slice], cmap="gray")
    ax.set_title(slice_cut + " normal, slice: " + str(slice))
    plt.show()


def test_gamma(img, gamma):
    # perform gamma transform
    img_gamma = tf.image.adjust_gamma(img, gamma=gamma, gain=1)

    return img.numpy(), img_gamma.numpy()


def test_brightness(img, mu=0.0, sigma=0.3):
    # perform brightness transform
    rnd_nb = np.random.normal(mu, sigma)
    img_b = tf.image.adjust_brightness(img, rnd_nb)

    return img, img_b.numpy()


def test_contrast(img, contrast_factor):
    # perform contrast transform
    img_cont = tf.image.adjust_contrast(img, contrast_factor=contrast_factor)

    return img, img_cont.numpy()

indexes = [20, 50, 80, 100, 200, 300, 400]
file, img_IDs = read_file()

for index in indexes:
    img, mask = read_get_image(file, img_IDs, index=index)
    gamma_list=[0.8, 0.9, 1.2, 1.4]
    b_list = [0.8]
    c_list = [0.3, 0.6, 0.9, 1.2, 1.5, 1.7]
    slice_cut = ['a']
    slice_list = range(50, 200, 25)
    ##transformations = ['gamma', 'brightness', 'contrast']
    transformations = ['gamma']

    for cut in slice_cut:
        for slice in slice_list:
            plot_img(mask, slice, cut, name="slice")

# if trans == 'brightness':
# for b in b_list:
# img, img_gamma = test_brightness(img)
# plot_img(img, img_gamma, b, slice, cut, trans)

# if trans == 'contrast':
# for c in c_list:
# img, img_gamma = test_contrast(img, c)
# plot_img(img, img_gamma, c, slice, cut, trans)
