import tensorflow as tf
from med_io.read_HD5F import *
import h5py
import numpy as np
from keras.utils.np_utils import to_categorical
import pickle

def sampling():
    pass

def convert_to_one_hot(mask_array):
    mask_one_hot = to_categorical(mask_array, num_classes=2)
    values_1 = np.unique(mask_one_hot[..., 0])
    values_2 = np.unique(mask_one_hot[..., 1])
    return mask_one_hot

def get_ids_none():
    path = './tests/test_IDs/elem_None.pickle'
    with open(path, 'rb') as handle:
        b = pickle.load(handle)
    return b['ids_None']

def read_Nones():
    list_ids_None = get_ids_none()
    rootdir_file = '/mnt/data/rawdata/Melanom/Tumorvolume'
    Data_Reader = HD5F_Reader(dataset='MELANOM', rootdir_file=rootdir_file)

    img_IDs = Data_Reader.img_IDs
    file = Data_Reader.file
    for id in list_ids_None:
        mask_array = file['mask'][id]
        mask_array = np.rollaxis(np.int32(np.array(mask_array)), 0, 4)
        mask_one_hot = convert_to_one_hot(mask_array)
        mask_values = np.unique(mask_array)
        print("values in array: ", mask_values)



def test_Sampling():
    rootdir_file = '/mnt/data/rawdata/Melanom/Tumorvolume'
    Data_Reader = HD5F_Reader(dataset='MELANOM', rootdir_file=rootdir_file)

    img_IDs = Data_Reader.img_IDs
    file = Data_Reader.file
    img = file['image']

    label_ax2_any = []
    pos_None = 0
    ids_None = []

    for index, id_im in enumerate(img_IDs):
        print("-------------------------------------------------------------------------------------------------")
        print("elem: ", index)

        #img_h5 = file['image'][id_im]
        #print("Shape of the image is: ", img_h5.shape)
        #print("Type of the image is: ", type(img_h5))
        # the form of the images are  (channel, H, W, D)

        mask_h5 = file['mask_iso'][id_im]  # mask or label
        print("Shape of the mask_h5 is: ", mask_h5.shape)
        print("Type of the mask is: ", type(mask_h5))

        #img_array = np.rollaxis(np.float32(np.array(img_h5)), 0, 4)
        #print("Shape of the image ARRAY is: ", img_array.shape)
        #print("Type of the image rolled is: ", type(img_array))

        mask_array = np.rollaxis(np.int32(np.array(mask_h5)), 0, 4)
        print("Shape of the mask ARRAY is: ", mask_array.shape)
        print("Type of the mask rolled is: ", type(mask_array))
        mask_one_hot = convert_to_one_hot(mask_array)
        mask_values = np.unique(mask_array)
        print("values in array: ", mask_values)

        label_ax2_any.append([np.any(mask_array[..., -1] == c, axis=2)
                                    for c in range(2)])
        label = mask_array[..., -1]
        valid_idx = np.argwhere(label_ax2_any[index][1] == True)
        print("valid_idx size: ", valid_idx.size)
        if valid_idx.size:
            # choose random valid position (2d)
            rnd = np.random.randint(0, valid_idx.shape[0])
            idx = valid_idx[rnd]
            # Sample additional index along the third axis(=2).
            # Voxel value should be equal to the class value.
            valid_idx = label[idx[0], idx[1], :]
            valid_idx = np.argwhere(valid_idx == 1)[0]
            rnd = np.random.choice(valid_idx)
            idx = [idx[0], idx[1], rnd]
        else:
            idx = None
            pos_None = pos_None+1
            ids_None.append(id_im)

        print("idx: ", idx)
        print("none positions until now: ", pos_None)

    print(ids_None)
    print(len(ids_None))
    path_test = './tests/test_IDs'
    if not os.path.exists(path_test): os.makedirs(path_test)
    with open(path_test + '/' + 'elem_None.pickle', 'wb') as fp:
        pickle.dump({'ids_None': ids_None}, fp, protocol=pickle.HIGHEST_PROTOCOL)

#test_Sampling()
#read_Nones()
