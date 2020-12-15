import h5py
import yaml
import pickle
from models.ModelSet import *
from tensorflow.keras.models import load_model
import tensorflow as tf
path = '../saved_models/training_Melanoma/model_U_net_melanoma/MELANOM.h5'
f = h5py.File(path, 'r')
print(list(f.keys()))
# will get a list of layer names which you can use as index
#d = f['dense']['dense_1']['kernel:0']
# <HDF5 dataset "kernel:0": shape (128, 1), type "<f4">
#d.shape == (128, 1)
#d[0] == array([-0.14390108], dtype=float32)

weights = f['model_weights']

## read the losses in test
scores = {}
file_losses = '../resulttraining_Melanoma/model_U_net_melanoma/evaluate_loss_and_metrics/MELANOM.pickle'

with open(file_losses, 'rb') as fp:
    p = pickle.Unpickler(fp)
    scores = p.load

with open(config['dir_dataset_info'] + '/max_shape_' + dataset + '.pickle', 'rb') as fp:
    config['max_shape'] = pickle.load(fp)

print("finisjed")