from tensorflow.keras.layers import Conv2D, Conv3D
from tensorflow.keras.layers import Conv2DTranspose, Conv3DTranspose
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import ReLU, LeakyReLU, PReLU, ELU, Softmax
from tensorflow.keras.layers import AveragePooling2D, AveragePooling3D
from tensorflow.keras.layers import MaxPooling2D, MaxPooling3D
from tensorflow.keras.layers import UpSampling2D, UpSampling3D
from tensorflow.keras.activations import tanh


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
        :return: result tensor
        """
        for item, item_param in zip(order, order_param):
            # Convolution
            if item == 'c' or item == 'dc':
                k_init = 'glorot_uniform'
                k_reg = None
                p = 'same'
                dr = (1,) * (len(x.shape) - 2)
                if item_param is not None:
                    k_init = item_param['kernel_initializer']
                    k_reg = item_param['kernel_regularizer']
                    p = item_param['padding']
                    dr = item_param['dilation_rate']
                    if order_priority:
                        c_f = item_param['filters']
                        c_k = item_param['kernels']
                        c_s = item_param['strides']
                    else:
                        c_f, c_k, c_s = f, k, s
                else:
                    c_f, c_k, c_s = f, k, s

                if item == 'c':
                    if len(x.shape) == 5:
                           x = Conv3D(c_f, c_k, c_s,
                                      padding=p,
                                      kernel_initializer=k_init,
                                      dilation_rate=dr,
                                      kernel_regularizer=k_reg)(x)
                    elif len(x.shape) == 4:
                        x = Conv2D(c_f, c_k, c_s,
                                   padding=p,
                                   kernel_initializer=k_init,
                                   dilation_rate=dr,
                                   kernel_regularizer=k_reg)(x)
                    else:
                        pass
                else:
                    if len(x.shape) == 5:
                        x = Conv3DTranspose(c_f, c_k, c_s,
                                            padding=p,
                                            kernel_initializer=k_init,
                                            dilation_rate=dr,
                                            kernel_regularizer=k_reg)(x)
                    elif len(x.shape) == 4:
                        x = Conv2DTranspose(c_f, c_k, c_s,
                                            padding=p,
                                            kernel_initializer=k_init,
                                            dilation_rate=dr,
                                            kernel_regularizer=k_reg)(x)
                    else:
                        pass
            # Normalization
            elif item == 'b':
                x = BatchNormalization()(x)
            # Activation
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

            # Pooling
            elif item == 'ap':  # average pooling
                ps = (2,) * (len(x.shape) - 2)
                st = None
                p = 'valid'
                df = None
                if item_param is not None:
                    ps = item_param['pool_size']
                    st = item_param['strides']
                    p = item_param['padding']
                    df = item_param['data_format']
                if len(x.shape) == 5: x = AveragePooling3D(ps, st, p, df)(x) 
                else: x = AveragePooling2D(ps, st, p, df)(x)
            elif item == 'mp':  # max pooling
                ps = (2,) * (len(x.shape) - 2)
                st = None
                p = 'valid'
                df = None
                if item_param is not None:
                    ps = item_param['pool_size']
                    st = item_param['strides']
                    p = item_param['padding']
                    df = item_param['data_format']
                if len(x.shape) == 5: x = MaxPooling3D(ps, st, p, df)(x)
                else: x = MaxPooling2D(ps, st, p, df)(x)
            elif item == 'up':  # up sampling
                if len(x.shape) == 5: x = UpSampling3D()(x)
                else: x = UpSampling2D()(x)
        return x
    return func
