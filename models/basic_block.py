import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.activations import tanh

"""basic network building blocks"""


def block(f=64, k=3, s=2, order=None, order_param=None, order_priority=False):

    """
    A block with one or more sequence layers.
    This block may contains one convolution/decovolution layer, or more covolution/decolution layers with same :param f,
     :param k, and :param s

    :param f: filters: type int: The dimensionality of the output space (i.e. the number of output filters in the convolution).
    :param k: kernel size: type int  or   tuple/list of 2or3 int: Size of the convolution windows
    :param s: strides: type int  or   tuple/list of 2or3 int: stride of convolution window
    :param order: type list of str: Order of layers in this block neural network:
                    c: convolution, 'dc': deconvolution, r: relu, b: batch normalization, p: prelu, e: elu,
                    's',softmax, 'ap':average pooling, 'mp':max pooling, 'up': up sampling

    :param order_param: type dict of dict: Parameter of :param order
    :param order_priority : type bool: True if bypassing  :param f, :param k, and :param s  , which are already
                                        determined by convolution parameter in :param  order_param
    :return: func: Model function
    """
    if order_param is None:
        order_param = [None] * 3
    if order is None:
        order = ['c', 'r', 'b']
    assert (len(order) == len(order_param))

    def func(x):
        """
        Function of sequence block by the order
        :param x: input tensor
        :return: result tensor        """
        for item, item_param in zip(order, order_param):

            # Convolution #
            if item == 'c' or item == 'dc':

                k_init, k_reg, p, dr = 'glorot_uniform', None, 'same', (1,) * (len(x.shape) - 2)
                if item_param is not None:
                    # :param item_param: dict, parameter for configuring convolution
                    k_init, k_reg, p, dr = item_param['kernel_initializer'], item_param['kernel_regularizer'], \
                                           item_param['padding'], item_param['dilation_rate']
                    if order_priority:
                        c_f, c_k, c_s = item_param['filters'], item_param['kernels'], item_param['strides']
                    else:
                        c_f, c_k, c_s = f, k, s
                else:
                    c_f, c_k, c_s = f, k, s

                if item == 'c':
                    if len(x.shape) == 5:
                           x = Conv3D(c_f, c_k, c_s, padding=p, kernel_initializer=k_init, dilation_rate=dr,
                                   kernel_regularizer=k_reg)(x)
                    elif len(x.shape) == 4:
                        x = Conv2D(c_f, c_k, c_s, padding=p, kernel_initializer=k_init, dilation_rate=dr,
                                   kernel_regularizer=k_reg)(x)
                    else:
                        pass
                else:

                    if len(x.shape) == 5:
                        x = Conv3DTranspose(c_f, c_k, c_s, padding=p, kernel_initializer=k_init, dilation_rate=dr,
                                            kernel_regularizer=k_reg)(x)
                    elif len(x.shape) == 4:
                        x = Conv2DTranspose(c_f, c_k, c_s, padding=p, kernel_initializer=k_init, dilation_rate=dr,
                                            kernel_regularizer=k_reg)(x)
                    else:
                        pass

            # Normalization #
            elif item == 'b':
                x = BatchNormalization()(x)

            # Activation #
            elif item == 'r':
                x = ReLU()(x)
            elif item == 'l':
                x = LeakyReLU(item_param['alpha'])(x)
            elif item == 'p':
                x = PReLU()(x)
            elif item == 'e':
                x = ELU(item_param['alpha'])(x)
            elif item == 's':
                x = Softmax()(x)
            elif item == 't':
                x = tanh()(x)

            # Pooling#
            elif item == 'ap': # average pooling
                ps, st, p, df = (2,) * (len(x.shape) - 2), None, 'valid', None
                if item_param is not None:
                    # :param item_param: dict, parameter for configuring average pooling
                    ps, st, p, df = item_param['pool_size'], item_param['strides'], item_param[
                        'padding'], item_param['data_format']
                    x = AveragePooling3D(ps, st, p, df)(x) if len(x.shape) == 5 else AveragePooling2D(ps, st, p, df)(x)

            elif item == 'mp': # max pooling
                ps, st, p, df = (2,) * (len(x.shape) - 2), None, 'valid', None
                if item_param is not None:
                    # :param item_param: dict, parameter for configuring pooling
                    ps, st, p, df = item_param['pool_size'], item_param['strides'], item_param[
                        'padding'], item_param['data_format']
                x = MaxPooling3D(ps, st, p, df)(x) if len(x.shape) == 5 else MaxPooling2D(ps, st, p, df)(x)

            elif item == 'up': # up sampling
                x = UpSampling3D()(x) if len(x.shape) == 5 else UpSampling2D()(x)

        return x

    return func
