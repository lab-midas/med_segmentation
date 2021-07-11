import numpy as np
import tensorflow as tf


def generate_label(config, label, patch_pos, patch_size):
    #
    """
    The function is es for network body_part_identification
    Generate labels for each patchs
    :param config: type dict: config parameters
    :param label: type 1d tf.Tensor or 1d list, the row line of the the body part boundary of each patient's image
    :param patch_pos: type 2d tf.Tensor or 2d list, the patch positions list
    :param patch_size: type 3d list: the patch size of input image
    :return:y_onehot: type  ndarray: predict label for each patch
    :return: y_reg: type  ndarray: absolute distance from the middle of the patch to the closest threshold between labels

    """

    def make_class_label(label, patch_pos, patch_size):
        '''
        Create one hot encoded class label for each patch
        1 for the label that the patch is closest to
        '''
        label=tf.sort(label, direction='DESCENDING')

        temp = [0]* patch_pos.shape[0]
        patch_pos = tf.cast(patch_pos, tf.float32)
        patch_size=tf.cast(patch_size,tf.float32)
        for idx in range(patch_pos.shape[0]):
            new_label = label*-1+(patch_pos[idx][2] + patch_size[2] // 2)
            class_pos = tf.where(new_label > 0)[0][-1]

            one_hot = tf.one_hot(class_pos, config['body_identification_n_classes'], dtype=tf.float32)
            temp[idx]=one_hot


        one_hot_label=tf.stack(temp, axis=0)

        return  one_hot_label

    def make_reg_label(label, patch_pos, patch_size):
        '''
        Return absolute distance from the middle of the patch
        to the closest threshhold between labels
        '''

        label = tf.sort(label, direction='DESCENDING')
        reg_label_=[]
        threshold_=[]

        for idx in range(config['body_identification_n_classes'] - 1):
            threshold_.append( (label[idx] + label[idx + 1]) / 2)

        threshold= tf.stack(threshold_,axis=0)

        for idx in range(patch_pos.shape[0]):
            temp = tf.abs(threshold - patch_pos[idx][2] - patch_size[2] / 2)
            reg_label_.append([tf.reduce_min(temp)])

        reg_label = tf.stack(reg_label_, axis=0)
        return reg_label

    y_onehot = make_class_label(label, patch_pos, patch_size)
    y_reg = make_reg_label(label, patch_pos, patch_size)

    return y_onehot ,y_reg
