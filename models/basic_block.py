import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.activations import tanh
from keras.regularizers import *
import tensorflow_addons as tfa
#from tensorflow_addons.layers import *

"""basic network building blocks"""


def block(f=64, k=3, s=2, order=None, order_param=None, order_priority=False, **kwargs):

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
    **kwargs: supported parameters: name
    :return: func: Model function
    """
    if order_param is None:
        order_param = [None] * 3
    if order is None:
        order = ['c', 'r', 'b']
    #print("len order param: ", len(order_param))
    #print("len order: ", len(order))
    #assert (len(order) == len(order_param))

    order_param_items = []
    activation_func = ['l', 'e']
    normalizations = ['g', 'b']
    pooling = ['mp', 'ap']
    for elem in order:
        if elem == 'c' or elem == 'dc':
            order_param_items.append(order_param['convolution'])

        elif elem in activation_func:
            order_param_items.append(order_param['activation_function'])

        elif elem in normalizations:
            order_param_items.append(order_param['normalization'])
        elif elem in pooling:
            order_param_items.append(order_param)
        else:
            order_param_items.append(None)

    #while len(order) < len(order_param):
        #new_elem = None
        #order.append(new_elem)
        #print("new len order param: ", len(order_param))
        #print("new len order: ", len(order))

    assert (len(order) == len(order_param_items))

    name = None
    dropout = None

    for key in kwargs.keys():

        if key == 'name':
            name = kwargs[key]

        if key == 'dropout':
            order.append('d')
            dropout_rate = kwargs[key]


    def func(x):
        """
        Function of sequence block by the order
        :param x: input tensor
        :return: result tensor        """

        def get_k_reg(k_reg):
            val = k_reg[1]
            if k_reg[0] == 'l2':
                func = l2(val)

            elif k_reg[0] == 'l1':
                func = l1(val)

            else:
                func = None

            return func

        for item, item_param in zip(order, order_param_items):

            # Convolution #
            if item == 'c' or item == 'dc':

                k_init, k_reg, p, dr = 'glorot_uniform', None, 'same', (1,) * (len(x.shape) - 2)
                bias_conv = False
                if item_param is not None:
                    # :param item_param: dict, parameter for configuring convolution
                    k_init, k_reg, p, dr, bias_conv = item_param['kernel_initializer'], get_k_reg(item_param['kernel_regularizer']), \
                                           item_param['padding'], item_param['dilation_rate'], item_param['bias']
                    if order_priority:
                        c_f, c_k, c_s = item_param['filters'], item_param['kernels'], item_param['strides']
                    else:
                        c_f, c_k, c_s = f, k, s
                else:
                    c_f, c_k, c_s = f, k, s

                if item == 'c':
                    if len(x.shape) == 5:
                           x = Conv3D(c_f, c_k, c_s, padding=p, kernel_initializer=k_init, dilation_rate=dr,
                                   kernel_regularizer=k_reg, use_bias=bias_conv)(x)
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
                
            elif item == 'g':
                param_norm = item_param['group_normalization']
                num_channels = f ## the number of output filters
                num_groups = param_norm['groups']

                # use only one group if the given number of groups is greater than the number of channels
                if num_channels < num_groups:
                    num_groups = 1

                assert num_channels % num_groups == 0, f'Expected number of channels in input ' \
                                                       f'to be divisible by num_groups. num_channels={num_channels}, ' \
                                                       f'num_groups={num_groups}'
                ## here it is assumed that axis = -1, channels are at last
                x = tfa.layers.GroupNormalization(num_groups, axis=-1, epsilon=float(param_norm['epsilon']))(x)

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
                    ps, st, p, df = item_param['pool_size'], item_param['strides'], \
                                    item_param['padding'], item_param['data_format']
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

            elif item == 'd': # dropout
                x = Dropout(dropout_rate)(x)

        return x

    return func

## basic block for fully connected NN

def block_FCN(hidden_layers=2, neurons_layer=[50, 100], activation=['r'], classification=True, classes=None):

    assert hidden_layers!=len(neurons_layer), "Number of hidden neurons per layer should be equal to number" \
                                              "of hidden layers"

    def func(x):

        if classification: ##if the network is used for classification

            ## observe the one hot encoding for the different classes
            for layer in range(hidden_layers):

                x = Dense(neurons_layer[layer], input_shape=(x.shape()), activation=activation)(x)

            ## now it is time to apply either softmax for multiclass

            x = Dense(classes, input_shape=(x.shape()), activation='softmax')(x)

            return x

        else: ## for regression

            ## observe the one hot encoding for the different classes
            for layer in range(hidden_layers):
                x = Dense(neurons_layer[layer], input_shape=(x.shape()), activation=activation)(x)

            ## now it is time to apply either softmax for multiclass

            x = Dense(1, input_shape=(x.shape()), activation=activation)(x)

            return x
