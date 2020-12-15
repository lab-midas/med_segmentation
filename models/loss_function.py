from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
import tensorflow.keras.backend as K
import tensorflow as tf


def l1_loss(y_true, y_pred, config):
    """ l1 loss
            y_true: true targets tensor.
            y_pred: predictions tensor.
    """

    sum_loss = 0
    for class_index in range(config['channel_label_num']):
        y_t = y_true[..., class_index]
        y_p = y_pred[..., class_index]
        sum_loss += K.mean((tf.abs(y_p - y_t))) * config['loss_channel_weight'][class_index]
    return sum_loss


def l2_loss(y_true, y_pred, config):
    sum_loss = 0
    for class_index in range(config['channel_label_num']):
        y_t = y_true[..., class_index]
        y_p = y_pred[..., class_index]
        sum_loss += K.mean(K.pow(y_p - y_t, 2)) * config['loss_channel_weight'][class_index]
    return sum_loss


def dice_loss(y_true, y_pred, config):
    """ Dice loss
            y_true: true targets tensor.
            y_pred: predictions tensor.
            Dice calculation with smoothing to avoid division by zero
    """
    # smooth = 1E-16
    smooth = K.epsilon()
    sum_loss, weight_sum = 0, 0
    for class_index in range(config['channel_label_num']):
        y_t = y_true[..., class_index] #(x, y, z)
        y_p = y_pred[..., class_index]
        intersection = K.sum(K.abs(y_t * y_p), axis=-1) #(x, y)
        loss = 1 - (2. * intersection + smooth) / (K.sum(K.square(y_t), -1) + K.sum(K.square(y_p), -1) + smooth)
        sum_loss += loss * config['loss_channel_weight'][class_index]
        weight_sum += config['loss_channel_weight'][class_index]
    return sum_loss / (weight_sum + smooth)

def dice_loss_melanoma(y_true, y_pred, config):
    """ Dice loss for Melanoma network
            y_true: true targets tensor.
            y_pred: predictions tensor.
            Dice calculation with smoothing to avoid division by zero
    """
    # smooth = 1E-16
    #assert y_true.shape == y_pred.shape
    smooth = K.epsilon()
    #assert len(y_true.shape) == 5
    sum_loss, weight_sum = 0, 0

    for class_index in range(config['num_classes']):
        y_t = y_true[..., class_index]
        y_p = y_pred[..., class_index]
        intersection = tf.math.reduce_sum(y_t * y_p) * config['loss_channel_weight'][class_index]
        denominator = tf.math.reduce_sum(y_t) + tf.math.reduce_sum(y_p) + smooth

        loss = 1 - (2. * intersection / denominator)

        sum_loss += loss ## this returns a tensor
        weight_sum += config['loss_channel_weight'][class_index] ## this returns a tensor too

    y_mean = sum_loss/weight_sum

    return y_mean

def dice_loss_melanoma_2(y_true, y_pred, config):
    """ Dice loss for Melanoma network
            y_true: true targets tensor.
            y_pred: predictions tensor.
            Dice calculation with smoothing to avoid division by zero
    """
    ## here it is assumed that the y_true is already in one hot encoded
    assert y_true.shape == y_pred.shape
    smooth = K.epsilon()
    sum_loss, weight_sum = 0, 0
    for class_index in range(config['num_classes']):
        y_t = y_true[..., class_index]
        y_p = y_pred[..., class_index]
        intersection = K.sum(K.abs(y_t * y_p), axis=-1)
        loss = 1 - (2. * intersection + smooth) / (K.sum(K.square(y_t), -1) + K.sum(K.square(y_p), -1) + smooth)
        sum_loss += loss * config['loss_channel_weight'][class_index] ## this returns a tensor
        weight_sum += config['loss_channel_weight'][class_index] ## this returns a tensor too

    return sum_loss / (weight_sum + smooth)

def dice_coefficient_loss(y_true, y_pred, config, smooth=K.epsilon(), axis=None):
    """ Dice coefficient along specific axis (same as  1+dice_loss() if axis=None)
            y_true: true targets tensor.
            y_pred: predictions tensor.
            smooth: smoothing parameter to avoid division by zero
            axis: along which to calculate Dice
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=axis)
    return -(2. * intersection + smooth) / (K.sum(K.abs(y_true), axis=axis) + K.sum(K.abs(y_pred), axis=axis) + smooth)


def dice_loss_v2(y_true, y_pred, config):
    smooth = 1E-16
    sum_loss, weight_sum = 0, 0
    for class_index in range(config['channel_label_num']):
        y_t = y_true[..., class_index]
        y_p = y_pred[..., class_index]
        intersection = K.sum(K.abs(y_t * y_p), axis=-1)
        loss = 1 - (2. * intersection + smooth) / (K.sum(K.abs(y_t)) + K.sum(K.abs(y_p)) + smooth)
        sum_loss += loss * config['loss_channel_weight'][class_index]
        weight_sum += config['loss_channel_weight'][class_index]
    return sum_loss / (weight_sum + smooth)


def jaccard_dist_loss_(y_true, y_pred):
    """ Jaccard distance loss
            y_true: true targets tensor.
            y_pred: predictions tensor.
    """
    smooth = K.epsilon()
    intersection = K.sum(K.abs(y_true * y_pred))
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred))
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return - jac


def focal_loss(y_true, y_pred, config, alpha=0.25, gamma=2.0):
    """ multi-class focal loss
            y_true: true targets tensor.
            y_pred: predictions tensor.
    """

    def sigmoid_focal_crossentropy(y_true, y_pred, alpha=0.25, gamma=2.0, from_logits=False):
        """ multi-class focal crossentropy
            y_true: true targets tensor.
            y_pred: predictions tensor.
            alpha: balancing factor.
            gamma: modulating factor.

            Weighted loss float `Tensor`. If `reduction` is `NONE`,this has the
            same shape as `y_true`; otherwise, it is scalar.
        """
        if gamma and gamma < 0:
            raise ValueError(
                "Value of gamma should be greater than or equal to zero")

        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.convert_to_tensor(y_true, dtype=y_pred.dtype)

        # Get the cross_entropy for each entry
        ce = K.binary_crossentropy(y_true, y_pred, from_logits=from_logits)

        # If logits are provided then convert the predictions into probabilities
        if from_logits:
            pred_prob = tf.sigmoid(y_pred)
        else:
            pred_prob = y_pred

        p_t = (y_true * pred_prob) + ((1 - y_true) * (1 - pred_prob))
        alpha_factor = 1.0
        modulating_factor = 1.0

        if alpha:
            alpha = tf.convert_to_tensor(alpha, dtype=K.floatx())
            alpha_factor = (y_true * alpha + (1 - y_true) * (1 - alpha))

        if gamma:
            gamma = tf.convert_to_tensor(gamma, dtype=K.floatx())
            modulating_factor = tf.pow((1.0 - p_t), gamma)

        # compute the final loss and return
        return tf.reduce_sum(alpha_factor * modulating_factor * ce, axis=-1)

    smooth = 1E-16
    sum_loss, weight_sum = 0, 0
    for class_index in range(config['channel_label_num']):
        fl = sigmoid_focal_crossentropy(y_true[..., class_index], y_pred[..., class_index], alpha=alpha, gamma=gamma)
        loss = K.sum(fl)
        sum_loss += loss * config['loss_channel_weight'][class_index]
        weight_sum += config['loss_channel_weight'][class_index]

    return sum_loss / (weight_sum + smooth)


def jaccard_dist_loss(y_true, y_pred, config):
    smooth = 1E-16
    sum_loss, weight_sum = 0, 0
    for class_index in range(config['channel_label_num']):
        y_t = y_true[..., class_index]
        y_p = y_pred[..., class_index]

        intersection = K.sum(K.abs(y_t * y_p))
        sum_ = K.sum(K.abs(y_t) + K.abs(y_p))
        loss = -(intersection + smooth) / (sum_ - intersection + smooth)
        sum_loss += loss * config['loss_channel_weight'][class_index]
        weight_sum += config['loss_channel_weight'][class_index]

    return sum_loss / (weight_sum + smooth)


def jaccard_dist_loss_hybrid(y_true, y_pred, config):
    smooth = 1E-16
    y_tru = y_true[0]
    y_pre = y_pred[0]
    print('y_tru', y_tru)
    intersection = K.sum(K.abs(y_tru * y_pre))
    sum_ = K.sum(K.abs(y_tru) + K.abs(y_pre))
    sum_loss = -(intersection + smooth) / (sum_ - intersection + smooth)

    y_tru = y_true[1]
    y_pre = y_pred[1]
    loss = K.mean((tf.abs(y_tru - y_pre)))
    sum_loss += loss

    return sum_loss


# ================
"""Loss function for Body identification """


def precision(y_true, y_pred):
    '''Metric: true positives / (true positives + false positives)'''

    neg_y_true = 1 - y_true
    tp = K.sum(y_true[...] * y_pred[...])
    fp = K.sum(neg_y_true[...] * y_pred[...])
    precision = tp / (tp + fp + K.epsilon())
    return precision


def sensitivity(y_true, y_pred):  # also called recall
    '''Metric: true positives / (true positives + false negatives)'''

    neg_y_pred = 1 - y_pred
    tp = K.sum(y_true[...] * y_pred[...])
    fn = K.sum(y_true[...] * neg_y_pred[...])
    sensitivity = tp / (tp + fn + K.epsilon())
    return sensitivity


def specificity(y_true, y_pred):
    '''Metric: true negatives / (true negatives + false positives)'''

    neg_y_true = 1 - y_true
    neg_y_pred = 1 - y_pred
    fp = K.sum(neg_y_true[...] * y_pred[...])
    tn = K.sum(neg_y_true[...] * neg_y_pred[...])
    specificity = tn / (tn + fp + K.epsilon())
    return specificity
