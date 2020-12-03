import random
from scipy import ndimage
import numpy as np
import tensorflow as tf


@tf.function
def rotate(volume):
    """Rotate the volume by a few degrees"""

    def scipy_rotate(volume):
        # define some rotation angles
        angles = [-20, -10, -5, 5, 10, 20]
        # pick angles at random
        angle = random.choice(angles)
        # rotate volume
        volume = ndimage.rotate(volume, angle, reshape=False)
        volume[volume < 0] = 0
        volume[volume > 1] = 1
        return volume

    augmented_volume = tf.numpy_function(scipy_rotate, [volume], tf.float32)
    return augmented_volume


def train_preprocessing(volume, label):
    """Process training data by rotating and adding a channel."""
    # Rotate volume
    volume = rotate(volume)
    volume = tf.expand_dims(volume, axis=3)
    return volume, label


def validation_preprocessing(volume, label):
    """Process validation data by only adding a channel."""
    volume = tf.expand_dims(volume, axis=3)
    return volume, label


def gamma_contrast(data_sample, gamma_range=(0.5, 2.0), invert_image=False, per_channel=False,
                   retain_stats=False):
    epsilon = 1e-7
    if invert_image:
        data_sample = - data_sample
    if not per_channel:

        if np.random.random() < 0.5 and gamma_range[0] < 1:
            gamma = np.random.uniform(gamma_range[0], 1)
        else:
            gamma = np.random.uniform(max(gamma_range[0], 1), gamma_range[1])

        minm = data_sample.min()
        rnge = data_sample.max() - minm

        data_sample = ((data_sample - minm) / float(rnge + epsilon))

        data_sample = tf.image.adjust_gamma(image=data_sample, gamma=gamma, gain=rnge) + minm
    else:
        for c in range(data_sample.shape[-1]):

            if np.random.random() < 0.5 and gamma_range[0] < 1:
                gamma = np.random.uniform(gamma_range[0], 1)
            else:
                gamma = np.random.uniform(max(gamma_range[0], 1), gamma_range[1])

            minm = data_sample[:, :, :, c].min()
            rnge = data_sample[:, :, :, c].max() - minm
            data_sample[c] = np.power(((data_sample[c] - minm) / float(rnge + epsilon)), gamma) * float(
                rnge + epsilon) + minm
    if invert_image:
        data_sample = - data_sample
    return data_sample


def brightness_transform(data_sample, mu=0.0, sigma=0.3, per_channel=True, p_per_channel=1):
    if not per_channel:
        rnd_nb = np.random.normal(mu, sigma)
        for c in range(data_sample.shape[-1]):
            if np.random.uniform() <= p_per_channel:
                ##data_sample[:, :, :, c] += rnd_nb
                data_sample = tf.image.adjust_brightness(data_sample, delta=rnd_nb)
    else:
        for c in range(data_sample.shape[-1]):
            if np.random.uniform() <= p_per_channel:
                rnd_nb = np.random.normal(mu, sigma)
                ##data_sample[:, :, :, c] += rnd_nb
                data_sample = tf.image.adjust_brightness(data_sample, delta=rnd_nb)
    return data_sample


def contrast_augmentation_transform(data_sample, contrast_range=(0.75, 1.25), preserve_range=True, per_channel=True):
    if not per_channel:
        mn = data_sample.mean()
        if preserve_range:
            minm = data_sample.min()
            maxm = data_sample.max()
        if np.random.random() < 0.5 and contrast_range[0] < 1:
            factor = np.random.uniform(contrast_range[0], 1)
        else:
            factor = np.random.uniform(max(contrast_range[0], 1), contrast_range[1])
        data_sample = (data_sample - mn) * factor + mn
        if preserve_range:
            data_sample[data_sample < minm] = minm
            data_sample[data_sample > maxm] = maxm
    else:
        for c in range(data_sample.shape[-1]):
            mn = data_sample[:, :, :, c].mean()
            if preserve_range:
                minm = data_sample[:, :, :, c].min()
                maxm = data_sample[:, :, :, c].max()
            if np.random.random() < 0.5 and contrast_range[0] < 1:
                factor = np.random.uniform(contrast_range[0], 1)
            else:
                factor = np.random.uniform(max(contrast_range[0], 1), contrast_range[1])
            data_sample[:, :, :, c] = (data_sample[:, :, :, c] - mn) * factor + mn
            if preserve_range:
                data_sample[:, :, :, c][data_sample[:, :, :, c] < minm] = minm
                data_sample[:, :, :, c][data_sample[:, :, :, c] > maxm] = maxm

    return data_sample
