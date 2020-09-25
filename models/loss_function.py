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

    smooth = 1E-16
    # smooth = K.epsilon()
    sum_loss, weight_sum = 0, 0
    for class_index in range(config['channel_label_num']):
        y_t = y_true[..., class_index]
        y_p = y_pred[..., class_index]
        intersection = K.sum(K.abs(y_t * y_p))
        loss = 1 - (2. * intersection + smooth) / (K.sum(K.square(y_t)) + K.sum(K.square(y_p)) + smooth)
        sum_loss += loss * config['loss_channel_weight'][class_index]
        weight_sum += config['loss_channel_weight'][class_index]
    return sum_loss / (weight_sum + smooth)


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

    smooth = K.epsilon()
    sum_loss, weight_sum = 0, 0
    for class_index in range(config['channel_label_num']):
        fl = sigmoid_focal_crossentropy(y_true[..., class_index], y_pred[..., class_index], alpha=alpha, gamma=gamma)
        loss = K.mean(fl)
        sum_loss += loss * config['loss_channel_weight'][class_index]
        weight_sum += config['loss_channel_weight'][class_index]

    return sum_loss / (weight_sum + smooth)

    # return sigmoid_focal_crossentropy(y_true, y_pred, alpha=alpha, gamma=gamma)


def jaccard_dist_loss(y_true, y_pred, config):
    smooth = K.epsilon()
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
    smooth = K.epsilon()
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
