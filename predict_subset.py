import os

import tensorflow as tf
import numpy as np
import nibabel as nib

from models.Premodel_Set import Premodel_Set
from med_io.get_pad_and_patch import get_predict_patches_index, unpatch_predict_image
from util import convert_integers_to_onehot, convert_onehot_to_integers
from plot.plot_config import plot_figures_single

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

name_model_1 = 'premodel_MRGE'
weights = '/' + os.path.join('mnt','data', 'projects', 'Segmentation',
                             'cnn_segmentation_results',
                             'weights_pretrained_models',
                             'AT_NAKO', 'mrge_final_FW0.hdf5')
path_input_data = '/mnt/share/rahauei1/NAKO_300_NIFTI'

# Take requested entries from 'mrge_final_FW0.yaml'
config = {
    'convolution_parameter': {'padding': 'same',
                              'num_parallel_calls': 1,
                              'kernel_regularizer': 'l2',
                              'dilation_rate': 1,
                              'kernel_initializer': 'he_normal'},
    'filters': 8,
    'patch_size': [32, 32, 32],
    'channel_img_num': 2,
    'channel_label_num': 4,
    'feed_pos': True,
    'pos_noise_stdv': 0.01,
    'loss_functions': {'dice_loss': 1},
    'multi_gpu': False,
    'custom_metrics': ['recall_all', 'recall_per_class'],
    'tensorflow_metrics': ['categorical_crossentropy'],
    'optimizer': {'name': 'Adam',
                  'args': {'learning_rate': 0.01}
                  },
    'use_multiple_loss_function': False,
    'multiple_loss_function': {'class': 'loss_function',
                               'reg': 'binary_crossentropy'},
    'plot_figure': ['plot_mosaic', 'plot_area_ratio'],
    'result_rootdir': '/mnt/share/rahauei1/Results',
    'model': 'pretrained_thomas_fw',
    'colormap': [[0, 0, 0.1], [0, 0.2, 0.8], [0, 1, 0.5], [0.8, 0.8, 0.8], [0.2, 0.1, 0.8], [0.8, 0.2, 0.1]],
    'display_origin_image_channel': 0,
    'plot_mosaic': {'slice_dim': 0,
                    'vspace': 2,
                    'hspace': 2,
                    'col': 5,
                    'alpha_origin': 0.6},
    'plot_area_ratio': {'slice_dim': 0,
                        'merge_channel_plot': False,
                        'plot_label_series': ['predict', 'label'],
                        'fig_size': [20, 40]},
    'select_image': {'plot_mosaic': 'predict_integers',
                     'plot_area_ratio': 'predict'},

    }

# Loading pretrained model from file as done in train.py
call_model = getattr(Premodel_Set, name_model_1)
model_1, _ = call_model(self=Premodel_Set, config=config)
model_1.summary()
model_1.load_weights(weights)

for patient in os.listdir(path_input_data):
    patient_dir = os.path.join(path_input_data, patient)
    # Load fat and water image
    fat_dir = os.path.join(patient_dir, 'fat')
    water_dir = os.path.join(patient_dir, 'water')
    img_fat = nib.load(os.path.join(fat_dir, '_'.join([patient, 'fat']) + '.nii.gz'))
    img_water = nib.load(os.path.join(water_dir, '_'.join([patient, 'water']) + '.nii.gz'))
    img_fat = np.array(img_fat.get_fdata())
    img_water = np.array(img_water.get_fdata())
    # Combine images into one tensor
    img_combined = np.stack((img_fat, img_water), axis=-1)
    # Patch image tensor for prediction
    img_patched, index_list = get_predict_patches_index(img_combined, config['patch_size'], overlap_rate=0.0)
    img_patched = np.array(img_patched)
    index_list_scaled = index_list / np.array([320, 260, 316])[..., 0]
    print(img_patched.shape, len(index_list_scaled))
    prediction_patched = model_1.predict(x=(img_patched, index_list_scaled), batch_size=1, verbose=1)
    # Unpatch prediction result
    prediction = unpatch_predict_image(prediction_patched, index_list, config['patch_size'], threshold=0.01)
    # Convert result as done in predict.py
    # TODO: What happens here - I do not get it
    predict_img_integers = convert_onehot_to_integers(prediction)
    predict_img_one_hot = convert_integers_to_onehot(predict_img_integers, num_classes=prediction.shape[-1])
    # Plotting as done in predict.py
    dict_data = {'predict_integers': predict_img_integers,
                 'predict_onehot': predict_img_one_hot,
                 'label_integers': None,
                 'label_onehot': None,
                 'original_image': img_combined,
                 'without_mask': np.zeros(predict_img_integers.shape)}
    plot_figures_single(config, dict_data, name_id=patient)
