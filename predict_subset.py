import os

from models.Premodel_Set import Premodel_Set
from models.ModelSet import ModelSet

name_model_1 = 'premodel_MRGE'
name_model_2 = 'model_MRGE'
weights = '/' + os.path.join('mnt','data', 'projects', 'Segmentation',
                             'cnn_segmentation_results',
                             'weights_pretrained_models',
                             'AT_NAKO', 'mrge_final_FW0.hdf5')

# Take requested entries from 'mrge_final_FW0.yaml'
config = {
    'convolution_parameter': {'padding': 'same',
                              'num_parallel_calls': 1,
                              'kernel_regularizer': ['l2', 0.01],
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
    }

# Loading pretrained model from file as done in train.py
call_model = getattr(Premodel_Set, name_model_1)
model_1, _ = call_model(self=Premodel_Set, config=config)
model_1.summary()
# Loading untrained model from scratch as done in train.py
call_model = getattr(ModelSet, name_model_2)
model_2, _ = call_model(self=ModelSet, config=config)
model_2.summary()
