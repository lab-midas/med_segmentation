import tensorflow as tf
import pickle
import yaml
from models.ModelSet import *
from models.loss_function import *
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
from tensorflow.keras.utils import multi_gpu_model
from models.blocks import *
import copy
from keras.utils.np_utils import to_categorical
from models.loss_torch import *
import os

def read_pickle_file(filename):
    objects = []
    with (open(filename, "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break
    return objects

def model_U_net_melanoma(config, inputs):
        ## is config the only parameter for the model?

    conv_param = config['convolution_parameter']
    conv_param_dilated = copy.deepcopy(conv_param)

    ## the input tensor of the application is generated
    #inputs = Input(shape=(*config['patch_size'],) + (config['channel_img_num'],), name='inp1')

    x = inputs
    print("Input shape of the network: ", x.shape)

    ## we design a 5 levels in the encoder path according to the described paper
    ## assume that config['filters_melanoma'] is 32 according to paper

    f_maps = [config['filters_melanoma'] * 2 ** i for i in range(config['number_of_levels'])]
    print("number of f maps used: ", f_maps)

    encoders = []
    list_f_maps = enumerate(f_maps)

    for i, out_feature_num in list_f_maps:
        if i == 0:
            encoder = encoder_block(out_feature_num, conv_kernel_size=(3,3,3), apply_pooling=False,
                                        pool_kernel_size=(2, 2, 2), basic_block=block_ExtResNet,
                                        conv_layer_order=['c', 'g', 'e'],
                                        order_param=conv_param, )

        elif i == len(f_maps)-1: ## last layer in encoder / bottleneck

            encoder = encoder_block(out_feature_num, conv_kernel_size=3, apply_pooling=True,
                                        pool_kernel_size=(2, 2, 2), pool_type='mp',
                                        basic_block=block_ExtResNet, conv_layer_order=['c', 'g', 'e'],
                                        order_param=conv_param, name='bottleneck')
            print("name: Bottleneck")
                #write_latent_space(encoder.get_layer.output)

        else:
            encoder = encoder_block(out_feature_num, conv_kernel_size=3, apply_pooling=True,
                                        pool_kernel_size=(2, 2, 2), pool_type='mp',
                                        basic_block=block_ExtResNet, conv_layer_order=['c', 'g', 'e'],
                                        order_param=conv_param)
                #print("middle: ", i)
        encoders.append(encoder)

    print("number of encoder paths: ", len(encoders))

    decoders = []
    reversed_f_maps = list(reversed(f_maps))

    for i in range(len(reversed_f_maps) - 1):

        if i == (len(reversed_f_maps) - 2): ## last decoder
            print("last decoder")
            decoder = decoder_block(reversed_f_maps[i + 1], kernel_size=(3, 3, 3),
                                        stride_factor=2,
                                        conv_layer_order=['c', 'g', 'e'],
                                        order_param=conv_param, last_decoder=True)

        else:
            decoder = decoder_block(reversed_f_maps[i + 1], kernel_size=(3, 3, 3),
                                        stride_factor=2, basic_module=block_ExtResNet,
                                        conv_layer_order=['c', 'g', 'e'],
                                        order_param=conv_param)
        decoders.append(decoder)

    print("number of decoder paths: ", len(decoders))

        ##--------------------------------------------------------------------------------------------------------
        ##-------------------------------------------------------------------------------------------------------

        # encoder part
    encoders_features = []
    num_encoder = 1
    for encoder in encoders:
        print("encoder num: ", num_encoder)
        x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
        encoders_features.insert(0, x)
        num_encoder = num_encoder+1

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

        #x = final_conv(f_maps[0]//2, kernel_size=3, conv_layer_order=['c', 'b', 'r'])(x)
        #print("number of labels: ", config['channel_label_num'])
    x = final_conv(2, kernel_size=(1,1,1), s=1,
                       conv_layer_order=['c', 'g', 'e'], order_param=conv_param)(x)

        ## here should be the softmax activation function

    x = block(order=['s'])(x)
    return x

def get_config(config_path):
    with open(config_path, "r") as yaml_file:
        config = yaml.load(yaml_file.read(), Loader=yaml.FullLoader)
    return config

def get_model(model='model_U_net_melanoma', config=None):
    call_model = getattr(ModelSet, model)
    model, list_metric_names = call_model(self=ModelSet, config=config)
    return model

def get_indexes(path_imgs):
    files = os.listdir(path_imgs)
    indexes = [file.split('_')[0] for file in files]
    return indexes

def read_info(index):
    path = 'tests/test_img/' + str(index) + '_elem.pickle'
    #path = 'tests/tests/test_img/' + str(index) + '_elem.pickle'
    with open(path, 'rb') as handle:
        b = pickle.load(handle)
    return b['image'], b['mask']

def test_loss_function():
    path_config = '../config/config_melanoma.yaml'
    path_imgs = 'tests/test_img/'
    config = get_config(path_config)
    indexes = get_indexes(path_imgs)
    for index in indexes:

        x, y_true = read_info(index)
        y_true_2 = to_categorical(y_true, num_classes=2)
        y_true_one_hot = tf.convert_to_tensor(y_true_2, dtype=tf.float32)
        y_pred = Softmax()(np.expand_dims(x, axis=0))
        #y_pred = model_U_net_melanoma(config=config, inputs=x)
        loss_tf = dice_loss_melanoma(y_true= np.expand_dims(y_true_one_hot, axis=0), y_pred=y_pred, config=config)
        y_o = np.rollaxis(np.expand_dims(y_true_2, axis=0), axis=4, start=1)
        y_p = np.rollaxis(y_pred.numpy(), axis=4, start=1)
        loss_torch = DiceLoss(y_p, y_o)
        print("loss tf " + str(index) + ": ", loss_tf)
        print("loss torch " + str(index) + ": ", loss_torch)

#for i in range(15):
 #   obj = test_loss_function()
  #  print("file read:", i)