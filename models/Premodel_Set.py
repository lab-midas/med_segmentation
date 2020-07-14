import tensorflow as tf
from tensorflow.keras.layers import *

import tensorflow.keras.backend as K
from .blocks import *
# from .loss_Function import *
import models.loss_function as loss_function
from .metrics import *
from util import convert_tf_optimizer
from tensorflow.keras import *
import copy
from models.ModelSet import *

"""
This is the collection of the pretrained models codes. In order to load the model correctly, The code is not changed. 
"""

class Premodel_Set:

    def premodel_MRGE(self,config,  multi=False, lbda=0, out_res=None, feed_pos=True, pos_noise_stdv=0):

        in_shape = (*config['patch_size'],) + (config['channel_img_num'],)
        print(in_shape)

        rls=[8, 4, 2, 1, 1]

        k_0=16

        def MR_local_path(mode, filters, initializer, lbda, padding='same'):
            # implement a normal residual path in a residual block, which is used as a path in the merge and run net
            # the path is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
            # bn -> relu -> conv

            return lambda x: Conv3D(filters,
                                    kernel_size=(3, 3, 3),
                                    strides=(1, 1, 1),
                                    padding=padding,
                                    kernel_initializer=initializer,
                                    kernel_regularizer=regularizers.l2(lbda))(
                Activation('relu')(BatchNormalization()(
                    Conv3D(filters,
                           kernel_size=(3, 3, 3),
                           strides=(1, 1, 1),
                           padding=padding,
                           kernel_initializer=initializer,
                           kernel_regularizer=regularizers.l2(lbda))(Activation('relu')(BatchNormalization()(x))))))

        def MR_global_path(mode, filters, dilation_rate, initializer, lbda, padding='same'):
            return lambda x: Conv3D(filters,
                                    kernel_size=(3, 3, 3),
                                    strides=(1, 1, 1),
                                    padding=padding,
                                    dilation_rate=dilation_rate,
                                    kernel_initializer=initializer,
                                    kernel_regularizer=regularizers.l2(lbda))(
                Activation('relu')(
                    BatchNormalization()(x)))  # remove bias regularizer, usually only weights needs to be regularized

        def MR_block_split(filters, lbda, initializer='he_normal', padding='same'):
            def MR_split_instance(x):
                x = Conv3D(filters=filters,
                           kernel_size=(1, 1, 1),
                           strides=(1, 1, 1),
                           padding=padding,
                           kernel_initializer=initializer,
                           kernel_regularizer=regularizers.l2(lbda))(
                    Activation('relu')(
                        BatchNormalization()(x)))
                x_out = y_out = x
                return x_out, y_out

            return MR_split_instance

        def MR_GE_block(mode, filters, dilation_rate, lbda, kernel_initializer='he_normal', padding='same'):
            # GE stands for global enhanced
            # a novel idea for combining local path with global path
            def MR_instance(x, y):
                mid = Add()([x, y])
                x_conv = MR_local_path(mode, filters, kernel_initializer, lbda, padding)(x)
                y_conv = MR_global_path(mode, filters, dilation_rate, kernel_initializer, lbda, padding)(y)
                x_out = Add()([x_conv, mid])
                y_out = Add()([y_conv, mid])
                return x_out, y_out

            return MR_instance

        def MR_GE_block_merge(mode, filters, dilation_rate, lbda, kernel_initializer='he_normal', padding='same'):
            def MR_merge_instance(x, y):
                mid = Add()([x, y])
                x_conv = MR_local_path(mode, filters, kernel_initializer, lbda, padding)(x)
                y_conv = MR_global_path(mode, filters, dilation_rate, kernel_initializer, lbda, padding)(y)
                out = Add()([Add()([x_conv, y_conv]), mid])
                return out

            return MR_merge_instance

        in_ = Input(shape=in_shape, name='input_X')

        in_pos = Input(shape=(3,), name='input_position')
        pos = Reshape(target_shape=(1, 1, 1, 3))(in_pos)
        if pos_noise_stdv != 0:
            pos = GaussianNoise(pos_noise_stdv)(pos)
        pos = BatchNormalization()(pos)

        shortcuts = []
        x = in_

        for l in rls:
            x, y = MR_block_split(k_0, lbda)(x)
            block_num = int(log2(l) + 1)
            rate_list = [2 ** i for i in range(block_num)]
            for rate in rate_list[:-1]:
                x, y = MR_GE_block('3D', filters=k_0, dilation_rate=rate, lbda=lbda)(x, y)
            x = MR_GE_block_merge('3D', filters=k_0, dilation_rate=rate_list[-1], lbda=lbda)(x, y)
            shortcuts.append(x)
            x = MaxPool3D()(x)
            k_0 = int(2 * k_0)

        k_0 = int(x.shape[-1])
        # add one dense conv at the bottleneck, shift the dense block for the decoder to make it symmetric
        x = Conv3D(filters=k_0,
                   kernel_size=(1, 1, 1),
                   strides=(1, 1, 1),
                   padding='same',
                   kernel_initializer='he_normal',
                   kernel_regularizer=regularizers.l2(lbda))(
            Activation('relu')(
                BatchNormalization()(x)))

        if feed_pos:
            shape = x.shape[1:4]
            pos = UpSampling3D(size=shape)(pos)
            x = Concatenate(axis=-1)([x, pos])

        for l, shortcut in reversed(list(zip(rls, shortcuts))):  # start from transpose conv then mrge
            x = Conv3DTranspose(filters=k_0, kernel_size=(3, 3, 3), strides=(2, 2, 2),
                                padding="same", kernel_initializer='he_normal',
                                kernel_regularizer=regularizers.l2(lbda))(x)
            x = Add()([shortcut, x])
            k_0 = int(k_0 // 2)
            x, y = MR_block_split(k_0, lbda)(x)
            block_num = int(log2(l) + 1)
            rate_list = [2 ** i for i in range(block_num)]
            for rate in rate_list[:-1]:
                x, y = MR_GE_block('3D', filters=k_0, dilation_rate=rate, lbda=lbda)(x, y)
            x = MR_GE_block_merge('3D', filters=k_0, dilation_rate=rate_list[-1], lbda=lbda)(x, y)

        x = Conv3D(filters=config['channel_label_num'],
                   kernel_size=(1, 1, 1),
                   strides=(1, 1, 1),
                   padding='same',
                   kernel_initializer='he_normal',
                   kernel_regularizer=regularizers.l2(lbda))(
            Activation('relu')(
                BatchNormalization()(x)))

        out = Activation('softmax', name='output_Y')(x)



        if feed_pos:
            return create_and_compile_model([in_, in_pos], out, config)
        else:
            return create_and_compile_model(in_, out, config)



    def premodel_UNet(self,config):
        """
        model_filename, channel_img,channel_label
        ATNew_UNet.hdf5 , 1 ,4
        NAKO_in_AT_UNet.hdf5, 2 ,4
        KORA_WBOrgan_UNet.h5 ,2 ,3
        NAKO_WBOrgan_UNet.h5, 2, 6
        :return:
        """

        input_shape = (*config['patch_size'],) + (config['channel_img_num'],)
        pool_size = (2, 2, 2)
        n_labels = config['channel_label_num']

        deconvolution = False
        depth = 4
        n_base_filters = 32

        batch_normalization = True
        activation_name = "sigmoid"

        def create_convolution_block(input_layer, n_filters, batch_normalization=False, kernel=(3, 3, 3),
                                     activation=None,
                                     padding='same', strides=(1, 1, 1), instance_normalization=False):

            layer = Conv3D(n_filters, kernel, padding=padding, strides=strides)(input_layer)
            if batch_normalization:
                layer = BatchNormalization(axis=1)(layer)
            elif instance_normalization:
                try:
                    from keras_contrib.layers.normalization import InstanceNormalization
                except ImportError:
                    raise ImportError("Install keras_contrib in order to use instance normalization."
                                      "\nTry: pip install git+https://www.github.com/farizrahman4u/keras-contrib.git")
                layer = InstanceNormalization(axis=1)(layer)
            if activation is None:
                return Activation('relu')(layer)
            else:
                return activation()(layer)

        def get_up_convolution(n_filters, pool_size, kernel_size=(2, 2, 2), strides=(2, 2, 2),
                               deconvolution=False):
            if deconvolution:
                return Deconvolution3D(filters=n_filters, kernel_size=kernel_size,
                                       strides=strides)
            else:
                return UpSampling3D(size=pool_size)

        inputs = Input(input_shape, name='input_X')
        current_layer = inputs
        levels = list()

        # add levels with max pooling
        for layer_depth in range(depth):
            layer1 = create_convolution_block(input_layer=current_layer, n_filters=n_base_filters * (2 ** layer_depth),
                                              batch_normalization=batch_normalization)
            layer2 = create_convolution_block(input_layer=layer1, n_filters=n_base_filters * (2 ** layer_depth) * 2,
                                              batch_normalization=batch_normalization)
            if layer_depth < depth - 1:
                current_layer = MaxPooling3D(pool_size=pool_size)(layer2)
                levels.append([layer1, layer2, current_layer])
            else:
                current_layer = layer2
                levels.append([layer1, layer2])

        # add levels with up-convolution or up-sampling
        for layer_depth in range(depth - 2, -1, -1):
            up_convolution = get_up_convolution(pool_size=pool_size, deconvolution=deconvolution,
                                                n_filters=current_layer.shape[4])(current_layer)
            concat = concatenate([up_convolution, levels[layer_depth][1]], axis=4)
            current_layer = create_convolution_block(n_filters=levels[layer_depth][1].shape[4],
                                                     input_layer=concat, batch_normalization=batch_normalization)
            current_layer = create_convolution_block(n_filters=levels[layer_depth][1].shape[4],
                                                     input_layer=current_layer,
                                                     batch_normalization=batch_normalization)

        final_convolution = Conv3D(n_labels, (1, 1, 1))(current_layer)
        act = Activation(activation_name, name='output_Y')(final_convolution)
        return create_and_compile_model(inputs, act, config)



    def premodel_DenseNet3D(self,config,feed_pos=True, pos_noise_stdv=0):
        in_shape = (*config['patch_size'],) + (config['channel_img_num'],)
        k = 16
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
            """Convolution Layer for DenseBlock."""

            return lambda x: Conv3D(filters=k,
                                    kernel_size=3 * (kernel_size,),
                                    padding='same',
                                    kernel_regularizer=regularizers.l2(lbda),
                                    bias_regularizer=regularizers.l2(lbda))(
                Activation('relu')(
                    BatchNormalization()(x)))

            # Transition Layers

        def transitionLayerPool(mode, f, lbda):
            """Transition Layer for encoder path."""

            return lambda x: AveragePooling3D(pool_size=3 * (2,))(
                denseConv('3D', f, 1, lbda)(x))

        def transitionLayerUp(mode, f, lbda):
            """Transition Layer for decoder path."""

            return lambda x: UpSampling3D()(
                denseConv('3D', f, 1, lbda)(x))

        in_ = Input(shape=in_shape, name='input_X')

        # add crop-position

        in_pos = Input(shape=(3,), name='input_position')
        pos = Reshape(target_shape=(1, 1, 1, 3))(in_pos)
        if pos_noise_stdv != 0:
            pos = GaussianNoise(pos_noise_stdv)(pos)
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
        if feed_pos:
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

        x = Conv3D(filters=3, kernel_size=(1, 1, 1))(x)
        out = Activation('softmax', name='output_Y')(x)


        if feed_pos:
            return create_and_compile_model([in_, in_pos], out, config)
        else:
            return create_and_compile_model(in_, out, config)



