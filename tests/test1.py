#import pytest
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
#from models.loss_function import *

from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
import tensorflow.keras.backend as K


def dice_loss_melanoma(y_true, y_pred):
    """ Dice loss for Melanoma network
            y_true: true targets tensor.
            y_pred: predictions tensor.
            Dice calculation with smoothing to avoid division by zero
    """
    # smooth = 1E-16
    loss_channel_weight = [1.0, 1.0]
    smooth = K.epsilon()
    sum_loss, weight_sum = 0, 0
    for class_index in range(2):
        y_t = y_true[..., class_index]
        print(y_t)
        y_p = y_pred[..., class_index]
        print(y_p)
        intersection = K.sum(K.abs(y_t * y_p), axis=-1)
        loss = 1 - (2. * intersection + smooth) / (K.sum(K.square(y_t), -1) + K.sum(K.square(y_p), -1) + smooth)
        sum_loss += loss * loss_channel_weight[class_index]
        weight_sum += loss_channel_weight[class_index]
    return sum_loss / (weight_sum + smooth)

def test_loss_function():
     tensor_test = tf.constant([[[[1, 2],[4, 5]], [[1, 2],[4, 5]], [[1, 2],[4, 5]]]
                                   ,[[[1, 2],[4, 5]], [[1, 2],[4, 5]], [[1, 2],[4, 5]]]])
     print(tensor_test)
     y_true = tf.constant([[[[6, 7],[4, 5]], [[6, 7],[4, 5]], [[6, 7],[4, 5]]]
                                   ,[[[7, 2],[7, 5]], [[7, 2],[7, 5]], [[1, 8],[2, 5]]]])
     print(y_true)
     dice_loss = dice_loss_melanoma(y_true, tensor_test)
     return dice_loss

def normalize(img, globalscale=False):

    print("Shape to normalize is: ", img.shape)

    if globalscale:
        maxval = np.amax(img)
        minval = np.amin(img)
        img = (img - minval) / (maxval - minval + 1E-16)


    else:
        img = [(img[..., i] - np.min(img[..., i])) / (np.ptp(img[..., i]) + 1E-16) for i in
                    range(img.shape[-1])]
        #img = np.rollaxis(np.float32(np.array(img)), 0, 4)

    print("Final Shape: ", img.shape)

    return img

def try_test():

    tensor_test = tf.constant([[[[1, 2], [3, 4]], [[1, 2], [3, 4]], [[1, 2], [3, 4]]]
                                  , [[[1, 2], [3, 4]], [[1, 2], [3, 4]], [[1, 2], [3, 4]]]])
    tensor_predict = tf.constant([[[[2, 3], [4, 5]], [[2, 3], [4, 5]], [[2, 3], [4, 5]]]
                                  , [[[2, 3], [4, 5]], [[2, 3], [4, 5]], [[2, 3], [4, 5]]]])

    print(tensor_test)
    print(tensor_predict)

    tensor_test_norm = normalize(tensor_test, globalscale=False)
    tensor_predict_norm = normalize(tensor_predict, globalscale=False)

    print(tensor_test_norm)
    print(tensor_predict_norm)

    mult_tensor = tensor_test_norm * tensor_predict_norm

    print(mult_tensor)

    #intersection = K.sum(K.abs(mult_tensor), axis=-1)


print(try_test())

