import tensorflow as tf
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
from tensorflow.keras.utils import multi_gpu_model
from .blocks import *
# from .loss_Function import *
import models.loss_function as loss_function
from .metrics import *
from util import convert_tf_optimizer
import copy


class ModelSet:
    """
    Model set which contains the network models
    """
    """
    === network models
    """
    # Merge-And-Run Mapping network
    def model_MRGE(self, config):

        conv_param_global = config['convolution_parameter']
        conv_param_local = copy.deepcopy(conv_param_global)
        conv_param_local['dilation_rate'] = 1
        in_pos = None

        filters = config['filters']
        input_shape = (*config['patch_size'],) + (config['channel_img_num'],)
        inputs = tf.keras.Input(shape=input_shape, name='inp1')

        shortcuts = []
        x = inputs

        # maximum dilation rate in each stage
        list_max_dilate_rate = [8, 4, 2, 1, 1]
        for l in list_max_dilate_rate:
            x, y = MR_block_split(filters, conv_param_local)(x)

            block_num = int(log2(l) + 1)
            rate_list = [2 ** i for i in range(block_num)]
            for rate in rate_list[:-1]:
                conv_param_global['dilation_rate'] = rate
                x, y = MR_GE_block(filters, conv_param_local,conv_param_global)(x, y)
            conv_param_global['dilation_rate'] = rate_list[-1]
            x = MR_GE_block_merge(filters, conv_param_local,conv_param_global)(x, y)
            shortcuts.append(x)
            x = MaxPool3D()(x)
            filters = int(2 * filters)

        filters = int(x.shape[-1])
        x = block(filters, 1, 1, order=['c'], order_param=[conv_param_local])(x)

        if config['feed_pos']:
            in_pos = Input(shape=(3,), name='input_position')
            pos = Reshape(target_shape=(1, 1, 1, 3))(in_pos)
            if config['pos_noise_stdv'] != 0:
                pos = GaussianNoise(config['pos_noise_stdv'])(pos)
            pos = BatchNormalization()(pos)
            pos = UpSampling3D(size=x.shape[1:4])(pos)
            x = Concatenate(axis=-1)([x, pos])

        for l, shortcut in reversed(list(zip(list_max_dilate_rate, shortcuts))):
            x = block(filters, 3, 2, order=['dc'], order_param=[conv_param_local])(x)
            x = Add()([shortcut, x])
            filters = int(filters // 2)
            x, y = MR_block_split(filters, conv_param_local)(x)
            rate_list = [2 ** i for i in range(int(log2(l) + 1))]
            for rate in rate_list[:-1]:
                conv_param_global['dilation_rate'] = rate
                x, y = MR_GE_block(filters, conv_param_local,conv_param_global)(x, y)

            conv_param_global['dilation_rate'] = rate_list[-1]
            x = MR_GE_block_merge(filters, conv_param_local,conv_param_global)(x, y)
            x = block(config['channel_label_num'], 1, 1, order=['b', 'r', 'c'], order_param=[None, None, conv_param_local])(x)

        out = Activation('softmax', name='output_Y')(x)

        if config['feed_pos']:
            return create_and_compile_model([inputs, in_pos], out, config)
        else:
            return create_and_compile_model(inputs, out, config)



    # Merge-And-Run network with
    def model_Dilated_Resnet(self, config):
        """An Alternative structure of MRGE, Simplify the MRGE block to a new block similar to ResNet"""

        conv_param_global = config['convolution_parameter']
        conv_param_local = copy.deepcopy(conv_param_global)
        conv_param_local['dilation_rate'] = 1
        in_pos = None

        filters = config['filters']
        input_shape = (*config['patch_size'],) + (config['channel_img_num'],)
        inputs = tf.keras.Input(shape=input_shape, name='inp1')
        shortcuts = []
        x = inputs
        # maximum dilation rate in each stage
        list_max_dilate_rate = [8, 4, 2, 1, 1]
        for max_dilate_rate in list_max_dilate_rate:
            dilate_rate = [2 ** i for i in range(int(log2(max_dilate_rate)) + 1)]
            x_list = [x]
            for rate in dilate_rate:
                conv_param_global['dilation_rate'] = rate
                x = block(filters, 3, 1, order=['c', 'r', 'b'], order_param=[conv_param_global, None, None])(x)
                x_list.append(x)
            x = tf.concat(x_list, axis=-1)
            x = block(filters, 3, 1, order=['c', 'r', 'b'], order_param=[conv_param_local, None, None])(x)
            shortcuts.append(x)
            x = MaxPool3D()(x)
            filters = int(2 * filters)

        filters = int(x.shape[-1])
        if config['feed_pos']:
            in_pos = Input(shape=(3,), name='input_position')
            pos = Reshape(target_shape=(1, 1, 1, 3))(in_pos)
            if config['pos_noise_stdv'] != 0:
                pos = GaussianNoise(config['pos_noise_stdv'])(pos)
            pos = BatchNormalization()(pos)
            pos = UpSampling3D(size=x.shape[1:4])(pos)
            x = Concatenate(axis=-1)([x, pos])

        x = block(filters * 4, 1, 1, order=['c', 'b', 'r'], order_param=[conv_param_local, None, None])(x)
        x = block(filters, 1, 1, order=['c', 'b', 'r'], order_param=[conv_param_local, None, None])(x)

        x = block(filters, 3, 2, order=['dc'], order_param=[conv_param_local])(x)
        for index_, (max_dilate_rate, shortcut) in enumerate(reversed(list(zip(list_max_dilate_rate, shortcuts)))):

            x = Add()([shortcut, x])
            filters = int(filters // 2)

            dilate_rate = [2 ** i for i in range(int(log2(max_dilate_rate)) + 1)]
            x_list = [x]
            for rate in dilate_rate:
                conv_param_global['dilation_rate'] = rate
                x = block(filters, 3, 1, order=['c', 'r', 'b'], order_param=[conv_param_global, None, None])(x)
                x_list.append(x)
            x = tf.concat(x_list, axis=-1)
            if index_ < len(list_max_dilate_rate) - 1:
                x = block(filters, 3, 2, order=['dc', 'r', 'b'], order_param=[conv_param_local, None, None])(x)

        x = block(filters, 3, 1, order=['c', 'r', 'b'], order_param=[conv_param_local, None, None])(x)
        out = block(config['channel_label_num'], 1, 1, order=['c', 'b', 's'], order_param=[conv_param_local, None, None])(x)

        if config['feed_pos']:
            return create_and_compile_model([inputs, in_pos], out, config)
        else:
            return create_and_compile_model(inputs, out, config)

    def model_position_densenet(self, config):
        conv_param = config['convolution_parameter']

        def denseBlock(dense_block_len, filters):

            def dense_block_instance(x):
                x1 = block(filters, 3, 1, order=['b', 'l', 'c'],
                           order_param=[None, None, conv_param])(x)
                x1 = block(filters, 3, 1, order=['b', 'l', 'c'],
                           order_param=[None, None, conv_param])(x1)
                ins = [x, x1]
                for i in range(dense_block_len):
                    temp_list = [s for s in ins]
                    x1 = block(filters, 3, 1, order=['b', 'l', 'c'],
                               order_param=[conv_param, None, None])(Concatenate(axis=-1)(ins))
                    x1 = block(filters, 3, 1, order=['b', 'r', 'c'],
                               order_param=[None, None, conv_param])(x1)
                    temp_list.append(x1)
                    ins = temp_list
                y = Concatenate(axis=-1)(ins)
                y = block(filters, 1, 1, order=['b', 'l', 'c'],
                          order_param=[None, None, conv_param])(y)
                return y

            return dense_block_instance

        inputs = Input(shape=(*config['patch_size'],) + (config['channel_img_num'],), name='inp1')
        x = inputs
        x = Conv3D(filters=config['filters'] * 2, kernel_size=(7, 7, 7), strides=(2, 2, 2), padding='same')(x)

        # filters = [config['filters'] * 2 ** i for i in range(5)]
        # filters = [config['filters']*2, config['filters']*2,config['filters']*2,config['filters']*4 ]
        filters = [8, 8, 8, 12]
        f2=[32,64,128,128]

        skip_layer = []

        dense_block_lens = [8, 8, 8, 8]
        for i, (f,ff) in enumerate(zip(filters,f2)):
            x = denseBlock(dense_block_lens[i], f)(x)
            skip_layer.append(x)
            x = block(ff, 3, 1, order=['b', 'r', 'c'],order_param=[None, None, conv_param])(x)
            x = AveragePooling3D()(x)
            print('x.shape', x.shape)

        # x = skip_layer[-1]

        list_pos = []
        if config['feed_pos']:
            pos_filters = [config['filters'] * 1 ** i for i in range(4, -1, -1)]
            in_pos = Input(shape=(3,), name='input_position')
            pos = Reshape(target_shape=(1, 1, 1, 3))(in_pos)
            if config['pos_noise_stdv'] != 0: pos = GaussianNoise(config['pos_noise_stdv'])(pos)
            # pos = UpSampling3D(size=x.shape[1:4])(BatchNormalization()(pos))
            x = Concatenate(axis=-1)([x, pos])

            for f in pos_filters:
                pos = block(f, 4, 2, order=['dc', 'b', 'r'], order_param=[conv_param, None, None])(pos)
                list_pos.append(pos)

        for index, (sk, f,ff) in enumerate(reversed(list(zip(skip_layer[:], filters[:], f2[:])))):

            x = denseBlock(dense_block_lens[index], f)(x)
            if index == 0:
                x = block(config['filters'], 2, 2, order=[ 'b', 'r','dc'], order_param=[None, None,conv_param])(x)
            else:
                x = block(ff, 3, 1, order=['b', 'r', 'c'], order_param=[None, None, conv_param])(x)
                x = UpSampling3D()(x)

            if config['feed_pos']:
                x = tf.concat([x, sk, list_pos[index]], axis=-1)
            else:
                x = tf.concat([x, sk], axis=-1)
        x = block(32, 3, 1, order=['b', 'r', 'c'], order_param=[None, None, conv_param])(x)
        x = UpSampling3D()(x)

        # x= block(config['filters'], 4, 2, order=[ 'b', 'r','dc'], order_param=[ None, None,conv_param])(x)
        out = block(config['channel_label_num'], 1, 1, order=['c', 'b', 's'], order_param=[conv_param, None, None])(x)

        if config['feed_pos']:
            return create_and_compile_model([inputs, in_pos], out, config)
        else:
            return create_and_compile_model(inputs, out, config)



    def model_dilated_dense_net(self, config, len_dense=None, base_filter=32, param_dense_filter=None):

        conv_param = config['convolution_parameter']
        in_pos = None

        def dense(x, f, rates, conv_param):
            """
            :param f: type int: number of filter
            :param rates: type list of int: list of positive ints
            :param len_dense: type list of int: list of positive ints
            :return:
            """

            for i, rate in enumerate(rates):
                conv_param['dilated_rate'] = rate
                x = block(f, 3, 1, order=['b', 'r', 'c'], order_param=[None, None, conv_param])(x)
            x = block(f, 3, 2, order=['b', 'r', 'c'], order_param=[None, None, conv_param])(x)
            return x

        if param_dense_filter is None: param_dense_filter = [1, 1]
        f1, f2 = param_dense_filter[0], param_dense_filter[1]

        inputs = Input(shape=(*config['patch_size'],) + (config['channel_img_num'],), name='inp0')
        dilation_rates = [[1, 1, 2, 2, 3], [1, 1, 3], [2, 1], [1, 1], [1, 1]]
        if len_dense == None: len_dense = [4, 4, 4, 4, 4]
        shortcuts = []
        x = inputs
        for rates, l in zip(dilation_rates, len_dense):
            x = dense_block(l, base_filter, conv_param)(x)
            shortcuts.append(x)
            x = dense(x, int(round((f1 + base_filter * l) * f2)), rates, conv_param)

        if config['feed_pos']:
            in_pos = Input(shape=(3,), name='input_position')
            pos = Reshape(target_shape=(1, 1, 1, 3))(in_pos)
            if config['pos_noise_stdv'] != 0: pos = GaussianNoise(config['pos_noise_stdv'])(pos)
            pos = UpSampling3D(size=x.shape[1:4])(BatchNormalization()(pos))
            x = Concatenate(axis=-1)([x, pos])


        conv_param['dilated_rate'] = 1
        for l, shortcut in reversed(list(zip(len_dense, shortcuts))):
            x = dense_block(l, base_filter, conv_param)(x)
            x = block(int(round((f1 + base_filter * l) * f2 / 2)), 3, 1, order=['b', 'r', 'c', 'up'],
                      order_param=[None, None, conv_param, None])(x)
            x = Concatenate(axis=-1)([shortcut, x])

        out = block(config['channel_label_num'], 4, 1, order=['c', 'b', 's'], order_param=[conv_param, None, None])(x)
        if config['feed_pos']:
            return create_and_compile_model([inputs, in_pos], out, config)
        else:
            return create_and_compile_model(inputs, out, config)
    # UNet with double decoder



    def model_U_net_double_decoder(self, config):

        "UNet with double and parallel decoder path "
        conv_param = config['convolution_parameter']
        conv_param_dilated = config['convolution_parameter']

        def denseBlock(dense_block_len, filters):

            def dense_block_instance(x):
                x1 = block(filters, 3, 1, order=['c', 'b', 'r'],
                           order_param=[conv_param, None, None])(x)
                x1 = block(filters, 3, 1, order=['c', 'b', 'r'],
                           order_param=[conv_param, None, None])(x1)
                ins = [x, x1]
                for i in range(dense_block_len):
                    temp_list = [s for s in ins]
                    x1 = block(filters, 3, 1, order=['c', 'b', 'r'],
                               order_param=[conv_param, None, None])(Concatenate(axis=-1)(ins))
                    x1 = block(filters, 3, 1, order=['c', 'b', 'r'],
                               order_param=[conv_param, None, None])(x1)
                    temp_list.append(x1)
                    ins = temp_list
                y = Concatenate(axis=-1)(ins)
                y = block(filters, 1, 1, order=['c', 'b', 'r'],
                          order_param=[conv_param, None, None])(y)
                return y

            return dense_block_instance

        inputs = Input(shape=(*config['patch_size'],) + (config['channel_img_num'],), name='inp1')
        x = inputs
        filters = [config['filters'] * 2 ** i for i in range(5)]
        filters_2 = [config['filters'] // 2 * 2 ** i for i in range(5)]
        skip_layer = []
        for index, f in enumerate(filters):
            conv_param['dilation_rate'] = 1
            x = block(f, 4, 2, order=['c', 'b', 'r'], order_param=[conv_param, None, None])(x)
            conv_param_dilated['dilation_rate'] = 2 if index < 4 else 1
            x = block(f, 4, 1, order=['c', 'b', 'r'], order_param=[conv_param_dilated, None, None])(x)
            skip_layer.append(x)
        x = skip_layer[-1]

        if config['feed_pos']:
            in_pos = Input(shape=(3,), name='input_position')
            pos = Reshape(target_shape=(1, 1, 1, 3))(in_pos)
            if config['pos_noise_stdv'] != 0: pos = GaussianNoise(config['pos_noise_stdv'])(pos)
            pos = UpSampling3D(size=x.shape[1:4])(BatchNormalization()(pos))
            #x = Concatenate(axis=-1)([x, pos])

        x_up = 0
        for index, (sk, f, f_2) in enumerate(reversed(list(zip(skip_layer[:-1], filters[:-1], filters_2[:-1])))):
            x = block(f, 4, 2, order=['dc', 'b', 'r'], order_param=[conv_param, None, None])(x)
            x_1 = x
            x = tf.concat([x, sk], axis=-1)
            if index == 0:
                x_up = block(f_2, 4, 2, order=['dc', 'b', 'r'], order_param=[conv_param, None, None])(x_1)
            else:
                x_1 = block(f_2, 4, 1, order=['c', 'b', 'r'], order_param=[conv_param, None, None])(x_1)
                x_1 = tf.concat([x_1, x_up], axis=-1)
                x_up = block(f_2, 4, 2, order=['dc', 'b', 'r'], order_param=[conv_param, None, None])(x_1)

        x = block(filters[0] // 2, 4, 2, order=['dc', 'b', 'r'], order_param=[conv_param, None, None])(x)
        x = x + x_up
        out = block(config['channel_label_num'], 4, 1, order=['c', 'b', 's'], order_param=[conv_param, None, None])(x)

        if config['feed_pos']:
            return create_and_compile_model([inputs, in_pos], out, config)
        else:
            return create_and_compile_model(inputs, out, config)



    def model_body_identification_hybrid(self, config):
        '''
        Model is build after Philip Wolfs (ISS master student) model
        Changed output shape and removed one dense layer at the end
        '''
        inputs = Input(shape=config['patch_size'], name='input_layer')
        n_base_filter = 32
        reshaped = Reshape([config['patch_size'][1], config['patch_size'][2], 1])(inputs)


        in_pos = Input(shape=(3,), name='input_position')
        # Some convolutional layers
        conv_1 = Conv2D(n_base_filter,
                        kernel_size=(2, 2),
                        padding='same',
                        activation='relu')(reshaped)
        conv_2 = Conv2D(n_base_filter,
                        kernel_size=(2, 2),
                        padding='same',
                        activation='relu')(conv_1)
        conv_2 = MaxPooling2D(pool_size=(3, 3), padding='same')(conv_2)

        # Some convolutional layers
        conv_3 = Conv2D(n_base_filter * 2,
                        kernel_size=(2, 2),
                        padding='same',
                        activation='relu')(conv_2)
        conv_4 = Conv2D(n_base_filter * 2,
                        kernel_size=(2, 2),
                        padding='same',
                        activation='relu')(conv_3)
        conv_4 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv_4)

        # Now layers 8-12 in Philips net, no pooling at the end
        conv_5 = Conv2D(n_base_filter * 4,
                        kernel_size=(2, 2),
                        padding='same',
                        activation='relu')(conv_4)
        conv_6 = Conv2D(n_base_filter * 8,
                        kernel_size=(2, 2),
                        padding='same',
                        activation='relu')(conv_5)
        conv_6 = MaxPooling2D(pool_size=(2, 2),
                              padding='same')(conv_6)

        conv_7 = Conv2D(n_base_filter * 16,
                        kernel_size=(2, 2),
                        padding='same',
                        activation='relu')(conv_6)

        conv_8 = Conv2D(n_base_filter * 32,
                        kernel_size=(2, 2),
                        padding='same',
                        activation='relu')(conv_7)

        # Decided against second dense layer,
        # As the dense/dense combination contributed another 16 million parameters
        #    dense_1 = Dense(4096, activation='relu')(conv_8)
        #    dropout_1 = Dropout(0.2)(dense_1)
        dense_2 = Dense(4096, activation='relu')(conv_8)

        # Here additional flattening layer to get right dimensionsionality
        flattening_1 = Flatten()(dense_2)
        dense_3 = Dense(config['body_identification_n_classes'], activation='relu')(flattening_1)
        landmark_class_probability = Dense(config['body_identification_n_classes'], activation='softmax', name='class')(dense_3)
        direct_regression = Dense(1, activation='linear', name='reg')(dense_3)

        # Wrap in a Model
        if config['feed_pos']:
            return create_and_compile_model([inputs, in_pos], (landmark_class_probability, direct_regression), config)
        else:
            return create_and_compile_model(inputs, [landmark_class_probability, direct_regression], config)


    def model_body_identification_classification(self, config):
        '''
        Model is build after Philip Wolfs (ISS master student) model
        Changed output shape and removed one dense layer at the end
        '''

        inputs = Input(shape=config['patch_size'], name='input_layer')
        n_base_filter = 32
        # reshaped = Reshape([config['patch_size'][1], config['patch_size'][2], 1])(inputs)

        # Some convolutional layers
        conv_1 = Conv2D(n_base_filter,
                        kernel_size=(2, 2),
                        padding='same',
                        activation='relu')(inputs)
        conv_2 = Conv2D(n_base_filter,
                        kernel_size=(2, 2),
                        padding='same',
                        activation='relu')(conv_1)
        conv_2 = MaxPooling2D(pool_size=(3, 3), padding='same')(conv_2)

        # Some convolutional layers
        conv_3 = Conv2D(n_base_filter * 2,
                        kernel_size=(2, 2),
                        padding='same',
                        activation='relu')(conv_2)
        conv_4 = Conv2D(n_base_filter * 2,
                        kernel_size=(2, 2),
                        padding='same',
                        activation='relu')(conv_3)
        conv_4 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv_4)

        # Now layers 8-12 in Philips net, no pooling at the end
        conv_5 = Conv2D(n_base_filter * 4,
                        kernel_size=(2, 2),
                        padding='same',
                        activation='relu')(conv_4)
        conv_6 = Conv2D(n_base_filter * 8,
                        kernel_size=(2, 2),
                        padding='same',
                        activation='relu')(conv_5)
        conv_6 = MaxPooling2D(pool_size=(2, 2),
                              padding='same')(conv_6)

        conv_7 = Conv2D(n_base_filter * 16,
                        kernel_size=(2, 2),
                        padding='same',
                        activation='relu')(conv_6)

        conv_8 = Conv2D(n_base_filter * 32,
                        kernel_size=(2, 2),
                        padding='same',
                        activation='relu')(conv_7)

        dense_2 = Dense(4096, activation='relu')(conv_8)

        # Here additional flattening layer to get right dimensionsionality
        flattening_1 = Flatten()(dense_2)
        outputs = Dense(config['body_identification_n_classes'], activation='softmax', name='output_a')(flattening_1)

        # Wrap in a Model
        if config['feed_pos']:
            return create_and_compile_model([inputs, in_pos], landmark_class_probability, config)
        else:
            return create_and_compile_model(inputs, outputs, config)



"""
=== end network models
"""




def create_and_compile_model(inputs, outputs, config,premodel=None):
    """
    create and compile model
    :param inputs: type Tensor: input of the network
    :param outputs: type Tensor: output of the network
    :param config: type dict: configuring parameter
    :return: model : type Model
    """

    def loss_func(y_true, y_pred):
        sum_ = 0
        if config['loss_functions'] is not None:
            for name_loss_function in config['loss_functions']:
                loss_func = getattr(loss_function, name_loss_function)(y_true, y_pred,config=config)
                weight = config['loss_functions'][name_loss_function]
                sum_ = sum_ + weight * loss_func
        return sum_
    if premodel is None:
        if config['feed_pos']:
            assert (len(inputs) >= 2)
            model = Model([inputs[0], inputs[1]], outputs)
            if config['multi_gpu']: model = multi_gpu_model(model, gpus=config['multi_gpu'])
        else:
            model = Model(inputs, outputs)
    else:
        model=premodel

    if config['multi_gpu']:  model = multi_gpu_model(model, gpus=config['multi_gpu'])
    flatten = lambda x: [y for l in x for y in flatten(l)] if type(x) is list else [x]

    if premodel is None:
        if isinstance(outputs,list):
            custom_metrics=[]
            for output in outputs:
                custom_metric = flatten(
                    [get_custom_metrics(output.shape[-1], m, config) for m in config['custom_metrics']])
                custom_metrics.append(custom_metric)
            custom_metrics=flatten(custom_metrics)
        else:
            custom_metrics = flatten([get_custom_metrics(outputs.shape[-1], m, config) for m in config['custom_metrics']])

    else:
        custom_metrics = flatten([get_custom_metrics(outputs.output_shape[-1], m, config) for m in config['custom_metrics']])
    list_metric_name = config['tensorflow_metrics'] + [m.__name__ for m in custom_metrics]
    optimizer_func = convert_tf_optimizer(config)

    if custom_metrics is None:
        custom_metrics_list=config['tensorflow_metrics']
    else:
        custom_metrics_list = config['tensorflow_metrics']+ custom_metrics

    if config['use_multiple_loss_function']:
        #  Multiple network output
        loss_func_dict= config['multiple_loss_function']
        for key in loss_func_dict.keys():
            if loss_func_dict[key]=='loss_function':
                loss_func_dict[key]=loss_func
    else:
        #  Single network output

        use_tensorflow_loss_function = config['use_tensorflow_loss_function']
        if not use_tensorflow_loss_function:
            loss_func_dict=loss_func
        else:
            print('use tensor loss')
            loss_func_dict=config['tensorflow_loss_function']

    model.compile(loss=loss_func_dict,
                  optimizer=optimizer_func,
                  metrics=custom_metrics_list)


    return model, list_metric_name
