import pickle
import numpy as np
import os

def read_info_patient(path, index):
    path_index = path + '/' + str(index) + '/info/info.pickle'
    # path = 'tests/tests/test_img/' + str(index) + '_elem.pickle'
    os.listdir(path)
    with open(path_index, 'rb') as handle:
        b = pickle.load(handle)
    return b

def read_dataset_info(path):
    with open(path, 'rb') as handle:
        b = pickle.load(handle)
    return b

def test_info_patient():
    indexes = ['3c52a5c495', '3c47ab2d69', '1be421b568']
    path = '../../data/tfrecords/Melanom/Tumorvolume'
    path_split = '../dataset_info/split_paths_MELANOM.pickle'
    path_max = '../dataset_info/max_shape_MELANOM.pickle'
    split = read_dataset_info(path_split)
    max_shape = read_dataset_info(path_max)
    print(split)
    print(max_shape)
    for index in indexes:
        info_patient = read_info_patient(path, index)
        print(info_patient.keys())
        print(info_patient)
        print("lecture done")

test_info_patient()