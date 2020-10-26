import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from math import log2
from .basic_block import *

"""combined network building blocks"""

"""Dilated convolution block"""


def DilatedConv(filters, conv_param):
    if conv_param['dilation_rate'] is None:
        print('Warning! dilation_rate is set to 1!')
    return block(filters, k=3, s=1, order=['b', 'r', 'c'], order_param=[None, None, conv_param])


"""atrous spatial pyramid pooling"""


def ASPP(filters, strides, conv_param, dilation_rate_list, image_level_pool_size):
    ## pyramid part
    def func(x):
        pyramid_1x1 = block(filters, 1, strides, order=['c', 'b', 'r'], order_param=[conv_param, None, None])(x)
        branch = [pyramid_1x1]
        for rate in dilation_rate_list:
            conv_param['dilation_rate'] = rate
            x = DilatedConv(filters, conv_param)(x)
            x = block(order=['b', 'r'], order_param=[None, None])(x)
            branch.append(x)

        ##image level part
        ap_param = {'pool_size': image_level_pool_size, 'padding': 'valid'}
        image_level_feature = block(filters, 1, order=['ap', 'c'], order_param=[ap_param, conv_param])
        image_level_feature = tf.compat.v1.image.resize_bilinear(images=image_level_feature, size=image_level_pool_size,
                                                                 align_corners=True)
        branch.append(image_level_feature)
        branch_logit = Concatenate(axis=-1)(branch)
        return block(filters, 1, strides, order=['c'], order_param=[conv_param])(branch_logit)

    return func


"""
=== blocks for Merge and Run (MR) series: MRGE_net, Focal_MRGE
"""


def MR_local_path(filters, conv_param):
    return block(filters, 3, 1, order=['b', 'r', 'c', 'b', 'r', 'c'],
                 order_param=[None, None, conv_param, None, None, conv_param])


def MR_global_path(filters, conv_param):
    # a novel idea, to include a global path in the merge and run net implemented with dilated conv
    return block(filters, k=3, s=1, order=['b', 'r', 'c'], order_param=[None, None, conv_param])


def MR_local_pr(filters, conv_param):
    def MR_local_pr(x):
        x = block(filters // 2, 1, 1, order=['c', 'r'], order_param=[conv_param, None])(x)
        x = block(filters // 2, 3, 1, order=['b', 'r', 'c', 'b', 'r', 'c'],
                  order_param=[None, None, conv_param, None, None, conv_param])(x)
        return block(filters, 1, 1, order=['c', 'r'], order_param=[conv_param, None])(x)

    return MR_local_pr


def MR_global_pr(filters, conv_param):
    def MR_global_pr(x):
        x = block(filters // 2, 1, 1, order=['c', 'r'], order_param=[conv_param, None])(x)
        x = block(filters, k=3, s=1, order=['b', 'r', 'c'], order_param=[None, None, conv_param])(x)
        return block(filters, 1, 1, order=['c', 'r'], order_param=[conv_param, None])(x)

    return MR_global_pr


def MR_local_pr_no_bn(filters, conv_param):
    def MR_local_pr_no_bn(x):
        x = block(filters // 2, 1, 1, order=['c', 'r'], order_param=[conv_param, None])(x)
        x = block(filters // 2, 3, 1, order=['c', 'r', 'c', 'r'],
                  order_param=[conv_param, None, conv_param, None])(x)
        return block(filters, 1, 1, order=['c', 'r'], order_param=[conv_param, None])(x)

    return MR_local_pr_no_bn


def MR_global_pr_no_bn(filters, conv_param):
    def MR_global_pr_no_bn(x):
        x = block(filters // 2, 1, 1, order=['c', 'r'], order_param=[conv_param, None])(x)
        x = block(filters // 2, 3, 1, order=['c', 'r', 'c', 'r'], order_param=[conv_param, None])(x)
        return block(filters, 1, 1, order=['c', 'r'], order_param=[conv_param, None])(x)

    return MR_global_pr_no_bn


def MR_block(filters, conv_param_local, conv_param_global):
    # implementation of the merge-and-run block in https://arxiv.org/pdf/1611.07718.pdf
    def MR_block(x, y):
        mid = Add()([x, y])
        x_conv = MR_local_path(filters, conv_param_local)(x)
        y_conv = MR_local_path(filters, conv_param_global)(y)
        x_out, y_out = Add()([x_conv, mid]), Add()([y_conv, mid])
        return x_out, y_out

    return MR_block


def MR_GE_block(filters, conv_param_local, conv_param_global):
    # GE stands for global enhanced
    # a novel idea for combining local path with global path
    def MR_GE_block(x, y):
        mid = Add()([x, y])
        x_conv = MR_local_path(filters, conv_param_local)(x)
        y_conv = MR_global_path(filters, conv_param_global)(y)
        x_out, y_out = Add()([x_conv, mid]), Add()([y_conv, mid])
        return x_out, y_out

    return MR_GE_block


def MR_block_split(filters, conv_param):
    def MR_block_split(x):
        x = block(filters, 1, 1, order=['b', 'r', 'c'], order_param=[None, None, conv_param])(x)
        x_out, y_out = x, x
        return x_out, y_out

    return MR_block_split


def MR_GE_block_merge(filters, conv_param_local, conv_param_global):
    def MR_GE_block_merge(x, y):
        mid = Add()([x, y])
        x_conv = MR_local_path(filters, conv_param_local)(x)
        y_conv = MR_global_path(filters, conv_param_global)(y)
        return Add()([Add()([x_conv, y_conv]), mid])

    return MR_GE_block_merge


def MRGE_exp_block(filters, dilation_max, conv_param_local, conv_param_global):
    def MRGE_exp_block(x):
        x, y = MR_block_split(filters, conv_param_local)(x)
        block_num = int(log2(dilation_max) + 1)
        rate_list = [2 ** i for i in range(block_num)]
        for rate in rate_list[:-1]:
            conv_param_local['dilation_rate'] = rate
            x, y = MR_GE_block(filters, conv_param_local, conv_param_global)(x, y)
        x = MR_GE_block_merge(filters, conv_param_local, conv_param_global)(x, y)
        return x

    return MRGE_exp_block


"""
=== end merge and run blocks
"""

"""
=== Residual networks: resnet, resnext blocks
"""


def resnext_branch(fin, fout, conv_param):
    def resnext_branch(x):
        x = block(fin, 1, 1, order=['c', 'r'], order_param=[conv_param, None])(x)
        x = block(fin, 3, 1, order=['b', 'r', 'c'], order_param=[None, None, conv_param])(x)
        return block(fout, 1, 1, order=['c', 'r'], order_param=[conv_param, None])(x)

    return resnext_branch


def resnext_block(sp, fin, fout, conv_param):
    def resnext_block(x):
        input = block(fout, 1, 1, order=['c', 'r'], order_param=[conv_param, None])(x)
        branch = [input]
        for i in range(sp):
            x = resnext_branch(fin, fout, conv_param)(input)
            branch.append(x)
        return Add()(branch)

    return resnext_block


def res_block(filters, conv_param, scale=0.1):
    def res_block(x):
        res = block(filters, 3, 1, order=['c', 'r', 'c'], order_param=[conv_param, None, conv_param])(x)
        res = Lambda(lambda x: x * scale)(res)
        return Add()([x, res])

    return res_block


###--------------------------------------------------------------------------------------------------------------------
##-------------------------------------------------------------------------------------------------------------------
## this block must contain 2 convolutions, residual connection
## the arqchitecture is described in the papes
##  https://arxiv.org/pdf/1706.00120.pdf


def block_ExtResNet(out_channels, kernel_size=3, order=['c', 'r', 'b'], name=None):
    def block_ExtResNet(x):

        print("dimension entering x is: ", x.shape)

        ## we start at creating 2 and the second output is used as residual connection
        conv1 = block(f=out_channels, k=kernel_size, s=1, order=['c', 'r', 'b'],
                      order_param=None, order_priority=False)(x)

        #print("dimension after conv 1 is: ", conv1.shape)

        residual = conv1

        conv2 = block(f=out_channels, k=kernel_size, s=1, order=['c', 'r', 'b'],
                      order_param=None, order_priority=False)(conv1)

        #print("dimension after conv 2 is: ", conv2.shape)

        ## remove non-linearity from the 3rd convolution since it's going to be applied after adding the residual
        activation_order = ['r', 'l', 'p', 'e', 's', 't']
        n_order = order

        for c in activation_order:
            if c in n_order:
                n_order.remove(c)

        conv3 = block(f=out_channels, k=kernel_size, s=1, order=n_order, order_param=None, order_priority=False)(conv2)

        #print("dimension after conv 3 is: ", conv3.shape)

        res_connection = tf.concat([conv3, residual], axis=-1)

        #print("dimension after residual connection is: ", res_connection.shape)

        ##now it is the non linearity applied
        #activation_order = ['r', 'l', 'p', 'e', 's', 't']
        n_order_act = order # default ['c', 'r', 'b']

        for elem in activation_order:
            if elem in n_order_act:
                n_order_act.remove(elem)

        x = block(f=out_channels, k=kernel_size, s=1, order=n_order_act,
                  order_param=None, order_priority=False, name=name)(res_connection)
        return x

    return block_ExtResNet


def encoder_block(out_channels, conv_kernel_size=3, apply_pooling=True,
                  pool_kernel_size=(2, 2, 2), pool_type='mp', basic_block=block_ExtResNet,
                  conv_layer_order=['c', 'r', 'b'], name=None):
    def encoder_block(x):

        ## here is missing the pool_kernel_size
        if apply_pooling:
            pooling_order =[]
            if pool_type == 'mp':
                pooling_order.append(pool_type)
                x = block(order=pooling_order)(x)

            else:
                pooling_order.append(pool_type)
                x = block(order=pooling_order)(x)

        x = basic_block(out_channels, kernel_size=conv_kernel_size,
                        order=conv_layer_order, name=name)(x)

        return x

    return encoder_block


def decoder_block(out_channels, kernel_size=3,
                  scale_factor=(2, 2, 2), basic_module=block_ExtResNet, pool_type='up',
                  conv_layer_order=['dc', 'r', 'b'], last_decoder=False):

    def decoder_block(x, encoder_feature):
        ##x = Conv3DTranspose(out_channels, kernel_size=kernel_size, strides=scale_factor, padding='same')(x)
        if not last_decoder:
            x = block(out_channels, kernel_size, s=2, order=conv_layer_order)(x)  # Deconvolution process
            x = tf.concat([x, encoder_feature], axis=-1)  # Concatenation with encoder part
            x = basic_module(out_channels, kernel_size=kernel_size, order=conv_layer_order)(x)

        else:
            x = block(out_channels, kernel_size, s=2, order=conv_layer_order)(x)  # Deconvolution process
            x = tf.concat([x, encoder_feature], axis=-1)  # Concatenation with encoder part
            x = block(out_channels, (5, 5, 1), s=1, order=conv_layer_order)(x)


        return x

    return decoder_block


###--------------------------------------------------------------------------------------------------------------------
###--------------------------------------------------------------------------------------------------------------------

"""
=== end residual networks
"""

"""
=== blocks for DCNet series: DCCN_ORI, DCCN_TR(transpose conv)
"""


# Dense connection
def dense_block(l, filters, conv_param):
    """
    :param l: type int: num of sub-block the dense block
    :param filters: type int: num of filters for dense convolution in each sub-block
    :param conv_param: type dict: parameters of convolution
    :return: function dense_block
    """

    def dense_block(x):
        ins = [x, dense_conv(filters, conv_param)(x)]
        for i in range(l - 1):
            ins.append(dense_conv(filters, conv_param)(tf.concat(ins, axis=-1)))
        return Concatenate(axis=-1)(ins)

    return dense_block


# Dense convolution
def dense_conv(filters, conv_param):
    """
    Sub-block of dense block
    :param filters:  type int: num of filters for convolution
    :param conv_param:   conv_param: type dict: parameters of convolution
    :return: function dense_conv
    """

    def dense_conv(x):
        x = block(filters, 3, 1, order=['b', 'r', 'c'], order_param=[None, None, conv_param])(x)
        return block(filters, 1, 1, order=['b', 'r', 'c'], order_param=[None, None, conv_param])(x)

    return dense_conv


# Transition pool layer
def transitionLayerPool(filters, conv_param):
    return lambda x: block(filters, 1, 1, order=['b', 'r', 'c', 'ap'],
                           order_param=[None, None, conv_param, {'pool_size': 2}])(x)


# Transition transpose up layer
def transitionLayerTransposeUp(filters, conv_param):
    def func(x):
        x = block(filters, 1, 1, order=['b', 'r', 'c'], order_param=[None, None, conv_param])(x)
        return block(filters, 3, 2, order=['dc'], order_param=[conv_param])(x)

    return func


# Attention layer
def attention_layer_1(filters_out, conv_param, filters=16, alpha=1):
    def func(x1, x2):
        x1 = block(filters, 1, 1, order=['c', 'b', 'r'], order_param=[conv_param, None, None])(x1)
        x2 = block(filters, 1, 1, order=['c', 'b', 'r'], order_param=[conv_param, None, None])(x2)
        x = block(filters, 1, 1, order=['r', 'b', 'c', 's'], order_param=[None, None, conv_param, None])(x1 + x2)
        return block(filters_out, 1, 1, order=['c', 'b', 'r'], order_param=[conv_param, None, None])(x * alpha * x1)

    return func


"""
=== end blocks 
"""
