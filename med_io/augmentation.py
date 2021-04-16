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


def gamma_contrast(data_sample, num_patches=324, num_channel=2, shape_data=None,
                   gamma_range=(0.5, 1.7), invert_image=False, per_channel=False,
                   retain_stats=False):
    epsilon = 1e-7
    data_sample_patch = []
    gamma_range_tensor = tf.convert_to_tensor(gamma_range)
    for patch in range(num_patches):

        if invert_image:
            data_sample = - data_sample
        if not per_channel:

            # if np.random.random() < 0.5 and gamma_range[0] < 1:
            #    gamma = np.random.uniform(gamma_range[0], 1)
            # else:
            # gamma = np.random.uniform(max(gamma_range[0], 1), gamma_range[1])
            def true_fn():
                gamma_fn = tf.random.uniform(shape=(), minval=gamma_range[0], maxval=1, seed=1)
                return gamma_fn

            def false_fn():
                gamma_fn = tf.random.uniform(shape=(), minval=tf.math.maximum(gamma_range[0], 1),
                                                         maxval=gamma_range[1], seed=1)
                return gamma_fn

            cond = tf.math.logical_and(tf.math.less(tf.random.uniform(shape=(), minval=0, maxval=0.99, seed=1), 0.5),
                                       tf.math.less(gamma_range_tensor[0], 1))

            gamma = tf.cond(cond, true_fn, false_fn)

            min_val_ten = tf.math.reduce_min(data_sample[patch, ...])
            range_tensor = tf.math.reduce_max(data_sample[patch, ...]) - min_val_ten
            data_sample_norm = tf.math.divide(tf.math.subtract(data_sample[patch, ...], min_val_ten),
                                              tf.math.add(range_tensor, epsilon))
            data_img = tf.image.adjust_gamma(image=data_sample_norm, gamma=gamma,
                                             gain=tf.math.add(range_tensor, epsilon))
            data_img = tf.math.add(data_img, min_val_ten)

            data_sample_patch.append(data_img)

        else:
            data_sample_per_channel = []
            for c in range(num_channel):

                def true_fn():
                    gamma_fn = tf.random_uniform_initializer(minval=gamma_range[0], maxval=1, seed=1)
                    return gamma_fn

                def false_fn():
                    gamma_fn = tf.random_uniform_initializer(minval=tf.math.maximum(gamma_range[0], 1),
                                                             maxval=gamma_range[1], seed=1)
                    return gamma_fn

                cond = tf.math.logical_and(tf.math.less(tf.random.uniform(shape=(), minval=0, maxval=0.99, seed=1), 0.5),
                                           tf.math.less(gamma_range_tensor[0], 1))

                gamma = tf.cond(cond, true_fn, false_fn)
                min_val_ten = tf.math.reduce_min(data_sample[patch, :, :, :, c])
                #rnge_tensor = tf.math.reduce_max(data_sample[patch, :, :, :, c]) - min_val_ten
                data_sample_norm = tf.math.divide(tf.math.subtract(data_sample[patch, ..., c], min_val_ten),
                                                  tf.math.add(range_tensor, epsilon))
                data_img = tf.image.adjust_gamma(image=data_sample_norm, gamma=gamma,
                                                 gain=tf.math.add(range_tensor, epsilon))
                data_img = tf.math.add(data_img, min_val_ten)
                data_sample_per_channel.append(data_img)

            data_sample_channel = tf.stack(data_sample_per_channel)
            data_sample_channel = tf.transpose(data_sample_channel, perm=[1, 2, 3, 0])
            data_sample_patch.append(data_sample_channel)

    data_sample_return = tf.stack(data_sample_patch)
    # data_sample_return = tf.transpose(data_sample_return, perm=[1, 2, 3, 4, 0])

    return data_sample_return


def brightness_transform(data_sample, mu=0.0, sigma=0.3, num_patches=324, num_channel=2,
                         shape_data=None, per_channel=True, p_per_channel=1):
    assert shape_data is not None, "Data shape should not be None"
    data_sample_patch = []
    for patch in range(num_patches):
        if not per_channel:
            data_sample_per_channel = []
            #rnd_nb = np.random.normal(mu, sigma)
            rnd_nb = tf.random.normal(shape=(), mean=mu, stddev=sigma, seed=1)
            rnd_nb = tf.cast(rnd_nb, dtype=tf.float32)
            #rnd_nb_tensor_1 = tf.multiply(tf.ones(shape=shape_data[:-1], dtype=tf.float32), rnd_nb)
            #rnd_nb_tensor = tf.convert_to_tensor(rnd_nb)
            for ch in range(num_channel):

                if np.random.uniform() <= p_per_channel:
                    sample_channel = tf.math.add(data_sample[patch, ..., ch], rnd_nb)
                    data_sample_per_channel.append(sample_channel)

            data_sample_channel = tf.stack(data_sample_per_channel)
            data_sample_channel = tf.transpose(data_sample_channel, perm=[1, 2, 3, 0])
            data_sample_patch.append(data_sample_channel)

        else:
            data_sample_per_channel = []
            for ch in range(num_channel):
                if np.random.uniform() <= p_per_channel:
                    #rnd_nb = np.random.normal(mu, sigma)
                    rnd_nb = tf.random.normal(shape=(), mean=mu, stddev=sigma, seed=1)
                    rnd_nb = tf.cast(rnd_nb, dtype=tf.float32)
                    #rnd_nb_tensor_1 = tf.multiply(tf.ones(shape=shape_data[:-1], dtype=tf.float32), rnd_nb)
                    #rnd_nb_tensor = tf.convert_to_tensor(rnd_nb)
                    sample_channel = tf.math.add(data_sample[patch, ..., ch], rnd_nb)
                    data_sample_per_channel.append(sample_channel)

            data_sample_channel = tf.stack(data_sample_per_channel)
            data_sample_channel = tf.transpose(data_sample_channel, perm=[1, 2, 3, 0])
            data_sample_patch.append(data_sample_channel)

    data_sample_return = tf.stack(data_sample_patch)
    # data_sample_return = tf.transpose(data_sample_return, perm=[1, 2, 3, 4, 0])

    return data_sample_return


def contrast_augmentation_transform(data_sample, contrast_range=(0.75, 1.25), num_patches=324, num_channel=2,
                                    shape_data=None, preserve_range=True, per_channel=True, p_per_sample=1):
    data_sample_patch = []
    for patch in range(num_patches):

        if not per_channel:
            mn = tf.math.reduce_mean(data_sample[patch, ...])

            if preserve_range:
                min_val_ten = tf.math.reduce_min(data_sample[patch, ...])
                max_val_tensor = tf.math.reduce_max(data_sample[patch, ...])

            #if np.random.random() < 0.5 and contrast_range[0] < 1:
            #    factor = np.random.uniform(contrast_range[0], 1)
            #else:
            #    factor = np.random.uniform(max(contrast_range[0], 1), contrast_range[1])
            def true_fn():
                factor_fn = tf.random.uniform(shape=(), minval=contrast_range[0], maxval=1, seed=1)
                return factor_fn

            def false_fn():
                factor_fn = tf.random.uniform(shape=(), minval=tf.math.maximum(contrast_range[0], 1),
                                                          maxval=contrast_range[1], seed=1)
                return factor_fn

            cond = tf.math.logical_and(tf.math.less(tf.random.uniform(shape=(), minval=0, maxval=0.99, seed=1), 0.5),
                                       tf.math.less(contrast_range[0], 1))

            factor = tf.cond(cond, true_fn, false_fn)

            data_sample_dif = tf.math.subtract(data_sample[patch, ...], mn)
            data_sample_mult = tf.math.multiply(data_sample_dif, factor)
            data_sample_channel = tf.math.add(data_sample_mult, mn)

            if preserve_range:
                ## know which positions are under the min value and over the max value
                ## the opposite operation is performed because just the False are going to be changed
                ## in the processing and the other values are to be retained

                min_bool = tf.math.greater(data_sample_channel, min_val_ten)
                max_bool = tf.math.less(data_sample_channel, max_val_tensor)

                data_sample_min_kept = tf.where(min_bool, data_sample_channel, min_val_ten)
                data_sample_channel = tf.where(max_bool, data_sample_min_kept, max_val_tensor)

            data_sample_patch.append(data_sample_channel)

        else:
            data_sample_per_channel = []
            for c in range(num_channel):
                mn = tf.math.reduce_mean(data_sample[patch, :, :, :, c])

                if preserve_range:
                    min_val_ten = tf.math.reduce_min(data_sample[patch, ..., c])
                    max_val_tensor = tf.math.reduce_max(data_sample[patch, ..., c])

                #if np.random.random() < 0.5 and contrast_range[0] < 1:
                #    factor = np.random.uniform(contrast_range[0], 1)
                #else:
                #    factor = np.random.uniform(max(contrast_range[0], 1), contrast_range[1])
                def true_fn():
                    factor_fn = tf.random.uniform(shape=(), minval=contrast_range[0], maxval=1, seed=1)
                    return factor_fn

                def false_fn():
                    factor_fn = tf.random.uniform(shape=(), minval=tf.math.maximum(contrast_range[0], 1),
                                                              maxval=contrast_range[1], seed=1)
                    return factor_fn

                cond = tf.math.logical_and(
                    tf.math.less(tf.random.uniform(shape=(), minval=0, maxval=0.99, seed=1), 0.5),
                    tf.math.less(contrast_range[0], 1))

                factor = tf.cond(cond, true_fn, false_fn)

                data_sample_dif = tf.math.subtract(data_sample[patch, ..., c], mn)
                data_sample_mult = tf.math.multiply(data_sample_dif, factor)
                data_sample_channel = tf.math.add(data_sample_mult, mn)

                if preserve_range:
                    min_bool = tf.math.greater(data_sample_channel, min_val_ten)
                    max_bool = tf.math.less(data_sample_channel, max_val_tensor)

                    data_sample_min_kept = tf.where(min_bool, data_sample_channel, min_val_ten)
                    data_sample_channel = tf.where(max_bool, data_sample_min_kept, max_val_tensor)

                data_sample_per_channel.append(data_sample_channel)

            data_sample_channel = tf.stack(data_sample_per_channel)
            data_sample_channel = tf.transpose(data_sample_channel, perm=[1, 2, 3, 0])
            data_sample_patch.append(data_sample_channel)

    data_sample_return = tf.stack(data_sample_patch)
    # data_sample_return = tf.transpose(data_sample_return, perm=[1, 2, 3, 4, 0])

    return data_sample_return
