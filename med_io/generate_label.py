import numpy as np
import tensorflow as tf
def generate_label(config, label, patch_pos, patch_size):
    # Special case for network body_identification
    # not tested
    def make_class_label(label, patch_pos, patch_size):
        '''
        Create one hot encoded class label for each patch
        1 for the label that the patch is closest to
        '''
        label=tf.sort(label, direction='DESCENDING')
        temp=[]
        for idx in range(patch_pos.shape[0]):
            new_label = (patch_pos[idx][2] + patch_size[2] // 2)-label
            class_pos = tf.where(new_label > 0)[0][-1]
            one_hot = tf.one_hot(class_pos,config['num_classes'],dtype=tf.float32)
            temp.append(one_hot)

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

        for idx in range(config['num_classes'] - 1):
            threshold_.append( (label[idx] + label[idx + 1]) / 2)

        threshold= tf.stack(threshold_,axis=0)

        for idx in range(patch_pos.shape[0]):
            temp = tf.abs(threshold - patch_pos[idx][2] - patch_size[2] / 2)
            reg_label_.append([tf.reduce_min(temp)])

        reg_label = tf.stack(reg_label_, axis=0)
        return reg_label

    y_onehot = make_class_label(label, patch_pos, patch_size)
    y_reg = make_reg_label(label, patch_pos, patch_size)

    return y_onehot,y_reg
