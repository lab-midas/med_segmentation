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
import numpy as np
from torch.autograd import Variable


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
                x, y = MR_GE_block(filters, conv_param_local, conv_param_global)(x, y)
            conv_param_global['dilation_rate'] = rate_list[-1]
            x = MR_GE_block_merge(filters, conv_param_local, conv_param_global)(x, y)
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
                x, y = MR_GE_block(filters, conv_param_local, conv_param_global)(x, y)

            conv_param_global['dilation_rate'] = rate_list[-1]
            x = MR_GE_block_merge(filters, conv_param_local, conv_param_global)(x, y)
            x = block(config['channel_label_num'], 1, 1, order=['b', 'r', 'c'],
                      order_param=[None, None, conv_param_local])(x)

        out = Activation('softmax', name='output_Y')(x)

        if config['feed_pos']:
            return create_and_compile_model([inputs, in_pos], out, config)
        else:
            return create_and_compile_model(inputs, out, config)

    # 3D DenseNet network
    def model_DenstNet_3D(self, config):
        conv_param_global = config['convolution_parameter']
        in_shape = (*config['patch_size'],) + (config['channel_img_num'],)

        k = 8
        ls = [8, 8, 8, 12]
        theta = 0.5
        k_0 = 32
        lbda = 0

        def denseBlock(mode, l, k, lbda):

            def dense_block_instance(x):
                ins = [x, denseConv('3D', k, 3, lbda)(
                    denseConv('3D', k, 1, lbda)(x))]
                for i in range(l - 1):
                    temp_list = [s for s in ins]
                    temp_list.append(denseConv('3D', k, 3, lbda)(
                        denseConv('3D', k, 1, lbda)(Concatenate(axis=-1)(ins))))
                    ins = temp_list
                y = Concatenate(axis=-1)(ins)
                return y

            return dense_block_instance

        def denseConv(mode, k, kernel_size, lbda):
            """Convolution Layer for DenseBlock.
            """
            return block(k, 3, 1, order=['b', 'r', 'c'])

            # Transition Layers

        def transitionLayerPool(mode, f, lbda):
            """Transition Layer for encoder path."""
            return block(f, 1, 1, order=['b', 'r', 'c', 'ap'])

        def transitionLayerUp(mode, f, lbda):
            """Transition Layer for decoder path."""
            return block(f, 1, 1, order=['b', 'r', 'c', 'up'])

        in_ = Input(shape=in_shape, name='input_X')

        # add crop-position


        in_pos = Input(shape=(3,), name='input_position')
        pos = Reshape(target_shape=(1, 1, 1, 3))(in_pos)
        if config['pos_noise_stdv'] != 0: pos = GaussianNoise(config['pos_noise_stdv'])(pos)


        pos = BatchNormalization()(pos)

        # encoder path
        x = Conv3D(filters=k_0, kernel_size=(7, 7, 7), strides=(2, 2, 2), padding='same')(in_)


        shortcuts = []
        for l in ls:
            x = denseBlock(mode='3D', l=l, k=k, lbda=lbda)(x)
            shortcuts.append(x)

            k_0 = int(round((k_0 + k * l) * theta))
            x = transitionLayerPool(mode='3D', f=k_0, lbda=lbda)(x)

        # concatenate position at feature map (bottleneck)

        if config['feed_pos']:
            shape = x.shape[1:4]
            pos = UpSampling3D(size=shape)(pos)
            x = Concatenate(axis=-1)([x, pos])


        # decoder path
        for l, shortcut in reversed(list(zip(ls, shortcuts))):
            x = denseBlock(mode='3D', l=l, k=k, lbda=lbda)(x)
            k_0 = int(round((k_0 + k * l) * theta / 2))
            x = transitionLayerUp(mode='3D', f=k_0, lbda=lbda)(x)
            x = Concatenate(axis=-1)([shortcut, x])
        x = UpSampling3D()(x)

        x = Conv3D(filters=config['channel_label_num'], kernel_size=(1, 1, 1))(x)
        out = Activation('softmax', name='output_Y')(x)


        if config['feed_pos']:
            return create_and_compile_model([in_, in_pos], out, config)
        else:
            return create_and_compile_model(in_, out, config)


    # modified 3D DenseNet
    def model_DenstNet_3Dv2(self, config):
        conv_param = config['convolution_parameter']

        def denseBlock(dense_block_len, filters):


            def dense_block_instance(x):
                x1 = block(filters, 1, 1, order=['b', 'r', 'c'], order_param=[None, None, None])(x)
                x1 = block(filters, 3, 1, order=['b', 'r', 'c'], order_param=[None, None, None])(x1)

                ins = [x, x1]
                for i in range(dense_block_len - 1):
                    temp_list = [s for s in ins]
                    x1 = block(filters, 1, 1, order=['b', 'r', 'c'], order_param=[None, None, None])(
                        Concatenate(axis=-1)(ins))
                    x1 = block(filters, 3, 1, order=['b', 'r', 'c'], order_param=[None, None, None])(x1)

                    temp_list.append(x1)
                    ins = temp_list
                y = Concatenate(axis=-1)(ins)
                return y

            return dense_block_instance

        inputs = Input(shape=(*config['patch_size'],) + (config['channel_img_num'],), name='inp1')
        x = inputs
        x = Conv3D(filters=35, kernel_size=(11, 11, 11), strides=(2, 2, 2), padding='same')(x)

        filters = [8, 8, 30, 64]
        f2 = [48, 64, 70, 128]
        f3 = [22, 30, 80, 128]

        skip_layer = []

        dense_block_lens = [8, 8, 8, 12]

        for i, (f, ff) in enumerate(zip(filters, f2)):
            x = denseBlock(dense_block_lens[i], f)(x)
            skip_layer.append(x)
            x = block(ff, 1, 1, order=['b', 'r', 'c'], order_param=[None, None, conv_param])(x)
            x = AveragePooling3D()(x)

        for index, (sk, f, ff) in enumerate(reversed(list(zip(skip_layer[:], filters[:], f3[:])))):
            x = denseBlock(dense_block_lens[index], f)(x)
            x = block(ff, 1, 1, order=['b', 'r', 'c'], order_param=[None, None, conv_param])(x)
            x = UpSampling3D()(x)
            x = tf.concat([x, sk], axis=-1)
        x = UpSampling3D()(x)
        x = Conv3D(filters=config['channel_label_num'], kernel_size=(1, 1, 1))(x)
        out = Activation('softmax', name='output_Y')(x)
        if config['feed_pos']:
            return create_and_compile_model([inputs, in_pos], out, config)
        else:
            return create_and_compile_model(inputs, out, config)

    # Dilated DenseNet
    def model_dilated_DenseNet(self, config, len_dense=None, base_filter=32, param_dense_filter=None):

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

    # UNet
    def model_U_net(self, config):

        conv_param = config['convolution_parameter']
        conv_param_d = copy.deepcopy(conv_param)

        inputs = Input(shape=(*config['patch_size'],) + (config['channel_img_num'],), name='inp1')
        x = inputs
        print(x.shape)

        x = block(4, 7, 1, order=['c', 'b', 'r'], order_param=[conv_param_d, None, None])(x)

        filters = [16, 32, 128, 128, 256]
        pos_filters = [128, 64, 32, 32, 16]
        skip_layer = []
        for i_f, f in enumerate(filters):
            if x.shape[1] > 2:
                x = block(f, 4, 2, order=['c', 'b', 'r'], order_param=[conv_param, None, None])(x)
                # x = MaxPool2D()(x)
            else:
                x = block(f, 2, 2, order=['c', 'b', 'r'], order_param=[conv_param, None, None])(x)
            skip_layer.append(x)
        x = skip_layer[-1]
        f = x.shape[-1]
        x = block(f * 4, 1, 1, order=['c', 'b', 'r'], order_param=[conv_param, None, None])(x)
        x = block(f, 1, 1, order=['c', 'b', 'r'], order_param=[conv_param, None, None])(x)

        list_pos = []

        if config['feed_pos']:
            in_pos = Input(shape=(3,), name='input_position')
            pos = Reshape(target_shape=(1, 1, 3))(in_pos)

            if config['pos_noise_stdv'] != 0: pos = GaussianNoise(config['pos_noise_stdv'])(pos)
            x = Concatenate(axis=-1)([x, pos])

            for f in pos_filters:
                pos = block(f, 4, 2, order=['dc', 'b', 'r'], order_param=[conv_param, None, None])(pos)
                list_pos.append(pos)
        skip_layer[-1] = x

        x = skip_layer[-2]
        f = x.shape[-1]
        x = block(f * 4, 2, 1, order=['c', 'b', 'r'], order_param=[conv_param, None, None])(x)
        x = block(f, 2, 1, order=['c', 'b', 'r'], order_param=[conv_param, None, None])(x)

        skip_layer[-2] = x
        x = skip_layer[-1]

        for index, (sk, f) in enumerate(reversed(list(zip(skip_layer[:-1], filters[:-1])))):
            x = block(f, 4, 2, order=['dc', 'b', 'r'], order_param=[conv_param, None, None])(x)

            x = tf.concat([x, sk, list_pos[index]], axis=-1)

        x = block(filters[0], 4, 2, order=['dc', 'b', 'r'], order_param=[conv_param, None, None])(x)
        # x = tf.concat([x, inputs], axis=-1)
        x = block(filters[0], 4, 1, order=['dc', 'b', 'r'], order_param=[conv_param, None, None])(x)
        out = block(config['channel_label_num'], 4, 1, order=['c', 'b', 's'], order_param=[conv_param, None, None])(x)

        if config['feed_pos']:
            return create_and_compile_model([inputs, in_pos], out, config)
        else:
            return create_and_compile_model(inputs, out, config)

    # body identification network for 2D coronal
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
        landmark_class_probability = Dense(config['body_identification_n_classes'], activation='softmax', name='class')(
            dense_3)
        direct_regression = Dense(1, activation='linear', name='reg')(dense_3)

        # Wrap in a Model
        if config['feed_pos']:
            return create_and_compile_model([inputs, in_pos], (landmark_class_probability, direct_regression), config)
        else:
            return create_and_compile_model(inputs, [landmark_class_probability, direct_regression], config)

    # body identification classification in 2D coronal
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

    # tumor lesion segmentation for PET/CT melanoma dataset
    def model_U_net_melanoma(self, config):
        ## is config the only parameter for the model?
        '''
        Model designed for melanom/methastases segmentation in PET/CT Images
        This model is based on the following paper:
        'https://arxiv.org/pdf/1706.00120.pdf'
        as a copy of 3D UNet created by Tobias Hepp
        '''

        conv_param = config['convolution_parameter']

        ## the input tensor of the application is generated
        inputs = Input(shape=(*config['patch_size'],) + (config['channel_img_num'],), name='inp1')

        x = inputs
        print("Input shape of the network: ", x.shape)

        ## we design a 5 levels in the encoder path according to the described paper
        ## assume that config['filters_melanoma'] is 32 according to paper

        f_maps = [config['filters_melanoma'] * 2 ** i for i in range(config['number_of_levels'])]
        print("number of f maps used: ", f_maps)

        ## config['filters_melanoma'] is assumed to be in config file, it can be added there
        ## config['number_of_levels'] are assumed to be in config file, it can be added there

        ##---------- U Net in encoder part ------------------------------------------------------------------------
        # for this experiment, we try with group normalization

        encoders = []
        list_f_maps = enumerate(f_maps)

        # the encoder parths are created and added to encoders list
        # An external Residual block is consider in the last encoder
        # without pooling, this is done to keep latent space
        # at the bottom for other networks

        for i, out_feature_num in list_f_maps:
            if i == 0:
                # feature maps at the end of convolution should be equal according to torch implementation

                encoder = encoder_block(out_feature_num, conv_kernel_size=(3, 3, 3), stride_size_conv=(1, 1, 1),
                                        apply_pooling=False, basic_block=block_ExtResNet,
                                        conv_layer_order=['c', 'g', 'e'],
                                        order_param=conv_param)


            elif i == len(f_maps)-1: ## last layer in encoder / bottleneck

                encoder = encoder_block(out_feature_num, conv_kernel_size=(3, 3, 3), stride_size_conv=(1, 1, 1),
                                        apply_pooling=True, stride_pool = (2, 2, 2),
                                        pool_kernel_size=(2, 2, 2), pool_type='mp',
                                        basic_block=block_ExtResNet, conv_layer_order=['c', 'g', 'e'],
                                        order_param=conv_param, name='bottleneck')
                print("name: Bottleneck")
                #write_latent_space(encoder.get_layer.output)

            else:
                encoder = encoder_block(out_feature_num, conv_kernel_size=(3, 3, 3), stride_size_conv=(1, 1, 1),
                                        apply_pooling=True, stride_pool = (2, 2, 2),
                                        pool_kernel_size=(2, 2, 2), pool_type='mp',
                                        basic_block=block_ExtResNet, conv_layer_order=['c', 'g', 'e'],
                                        order_param=conv_param)
            encoders.append(encoder)

        print("number of encoder paths: ", len(encoders))

        ##---------------------------decoder part-----------------------------------------------
        decoders = []
        reversed_f_maps = list(reversed(f_maps))

        for i in range(len(reversed_f_maps) - 1):

            if i == (len(reversed_f_maps) - 2): ## last decoder
                print("last decoder")
                decoder = decoder_block(reversed_f_maps[i + 1], kernel_size=(3, 3, 3), stride_size_conv=(1, 1, 1),
                                        stride_factor_up=(2, 2, 2), basic_module=block_ExtResNet,
                                        conv_layer_order=['c', 'g', 'e'],
                                        order_up=['dc'], order_param=conv_param,
                                        last_decoder=True, concat=False)

            else:
                decoder = decoder_block(reversed_f_maps[i + 1], kernel_size=(3, 3, 3), stride_size_conv=(1, 1, 1),
                                        stride_factor_up=(2, 2, 2), basic_module=block_ExtResNet,
                                        conv_layer_order=['c', 'g', 'e'],
                                        order_up=['dc'], order_param=conv_param,
                                        concat=False)

            decoders.append(decoder)

        print("number of decoder paths: ", len(decoders))

        # join encoder and decoder in a Unet architecture
        encoders_features = []
        num_encoder = 1
        for encoder in encoders:
            print("encoder num: ", num_encoder)
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)
            num_encoder = num_encoder+1

        # remove the last encoder's output from the list
        # !!remember: the first in the list
        encoders_features = encoders_features[1:]

        num_decoder = 1
        for decoder, encoder_feature in zip(decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            print("decoder num: ", num_decoder)
            x = decoder(x, encoder_feature)
            num_decoder = num_decoder + 1

        ## we have another final convolution according to the architecture proposed
        ##final_conv

        x = final_conv(2, kernel_size=(1,1,1), s=1,
                       conv_layer_order=['c'], order_param=conv_param)(x)

        ## here should be the softmax activation function, sigmoid can also be used

        x = block(order=['s'])(x)

        return create_and_compile_model(inputs, x, config)


"""
=== end network models
"""

def create_and_compile_model(inputs, outputs, config, premodel=None):
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
                loss_func = getattr(loss_function, name_loss_function)(y_true, y_pred, config=config)

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
        model = premodel

    if config['multi_gpu']:  model = multi_gpu_model(model, gpus=config['multi_gpu'])
    flatten = lambda x: [y for l in x for y in flatten(l)] if type(x) is list else [x]

    if premodel is None:
        if isinstance(outputs, list):
            custom_metrics = []
            for output in outputs:
                custom_metric = flatten(
                    [get_custom_metrics(output.shape[-1], m, config) for m in config['custom_metrics']])
                custom_metrics.append(custom_metric)
            custom_metrics = flatten(custom_metrics)
        else:
            custom_metrics = flatten(
                [get_custom_metrics(outputs.shape[-1], m, config) for m in config['custom_metrics']])

    else:
        custom_metrics = flatten(
            [get_custom_metrics(outputs.output_shape[-1], m, config) for m in config['custom_metrics']])
    list_metric_name = config['tensorflow_metrics'] + [m.__name__ for m in custom_metrics]
    optimizer_func = convert_tf_optimizer(config)

    if custom_metrics is None:
        custom_metrics_list = config['tensorflow_metrics']
    else:
        custom_metrics_list = config['tensorflow_metrics'] + custom_metrics

    if config['use_multiple_loss_function']:
        #  Multiple network output
        loss_func_dict = config['multiple_loss_function']
        for key in loss_func_dict.keys():
            if loss_func_dict[key] == 'loss_function':
                loss_func_dict[key] = loss_func
    else:
        #  Single network output


        use_tensorflow_loss_function = config['use_tensorflow_loss_function']
        if not use_tensorflow_loss_function:
            loss_func_dict = loss_func
        else:
            print('use tensor loss')
            loss_func_dict = config['tensorflow_loss_function']


    print("Start compiling model")
    model.compile(loss=loss_func_dict,
                  optimizer=optimizer_func,
                  metrics=custom_metrics_list)


    return model, list_metric_name

