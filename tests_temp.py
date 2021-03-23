import pickle
import h5py
"""
This is a collection of python code used for various test in my Bachelors
Thesis. This code is only supposed to be temporary and should not be merged into
master...
"""


def drop_ids_from_hdf5(patches_info_filepath, hdf5_filepath):
    """
    patches_info_filepath: path to pickle file with patches info
    hdf5_filepath: path to hdf5 file
    """
    with open(patches_info_filepath, 'rb') as f:
        patches_info = pickle.load(f)
    patches_to_keep = patches_info['init_ids'] + patches_info['val_ids']

    i = 0
    with h5py.File(hdf5_filepath, 'r+') as hdf5_file:
        print(hdf5_file.keys())
        print('Num of images: {0}'.format(len(hdf5_file['images'].keys())))
        for key in hdf5_file['images']:
            if key.encode('ascii') not in patches_to_keep:
                i += 1
                del hdf5_file['images'][key]
                del hdf5_file['labels'][key]

        print('dropped {0} images'.format(i))
        print('Num of images left: {0}'.format(len(hdf5_file['images'].keys())))
        # check if all wanted patches are still in hdf5 file
        check = True
        for key in patches_to_keep:
            if key not in hdf5_file['images'].keys():
                check = False
        print(check)


