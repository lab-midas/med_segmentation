""" This file contains code to analyse the patches used in the AL process using
    the patches data itself (hdf5 file) and the info of the origin of the data
    saved during the training process
    Plotting performed separately (in my case in plot_csv_data/plot_patches_info.py
    on my local computer, I plan to put the code in the container as well)"""
import pickle
from pathlib import Path
import h5py
import datetime
import numpy as np
import pandas as pd


# short quick way of runnig the analysis
def run():
    hdf5_path = './patches_data_hdf5/al_NAKO_AT_uniform_distribution-28-02-21.hdf5'
    hdf5_info_path = './patches_data_hdf5/al_NAKO_AT_uniform_distribution-28-02-21-patches_info.pickle'
    patches_infos_paths = ['./result/patches_info/AL_Exp9-1_NAKO_AT-patches_info.pickle',
                           './result/patches_info/AL_Exp9-2_NAKO_AT-patches_info.pickle',
                           './result/patches_info/AL_Exp9-3_NAKO_AT-patches_info.pickle',
                           './result/patches_info/AL_Exp9-4_NAKO_AT-patches_info.pickle',
                           './result/patches_info/AL_Exp9-5_NAKO_AT-patches_info.pickle',
                           './result/patches_info/AL_Exp9-6_NAKO_AT-patches_info.pickle',
                           './result/patches_info/AL_Exp9-7_NAKO_AT-patches_info.pickle',
                           './result/patches_info/AL_Exp9-8_NAKO_AT-patches_info.pickle',
                           './result/patches_info/AL_Exp9-9_NAKO_AT-patches_info.pickle']

    analyse_patches_main(hdf5_path, hdf5_info_path, patches_infos_paths, 'init_ids')


def analyse_patches_main(hdf5_path, hdf5_info_path, patches_infos_paths, id_set='init_ids'):
    """
        hdf5_path
        path to pickle file containing info on origin of patches
        list of patches_infos_paths (pickle files) as saved in training process
        key to list of ids to analyse in patches_infos file e.g. 'init_ids'
    """
    with open(hdf5_info_path, 'rb') as f:
        hdf5_info = pickle.load(f)

    # DataFrames where results will be stored
    patient_counts = {}
    class_counts = {}
    save_basedir = Path('../Experiments/Analysis', id_set)
    save_basedir.mkdir(parents=True, exist_ok=True)

    # Perform analysis for all stated patches_info pickle files (Experiments)
    for path in patches_infos_paths:
        # load lists with used ids from patches_infos files
        with open(path, 'rb') as f:
            patches_infos = pickle.load(f)
        print('hdf5_file used in analysis:{0}, file stated in patches '
              'info:{1}'.format(hdf5_path, patches_infos['hdf5_file']))

        # select which set of patches to analyse (e.g. init_ids)
        patches_to_analyse = patches_infos[id_set]

        # analyse:
        print('Count patients')
        patient_counts[path] = count_patients(hdf5_info, patches_to_analyse)
        print('Count classes')
        class_counts[path] = count_classes(hdf5_path, patches_to_analyse)

    # save results for later analysis and plotting
    path_patient_count = save_basedir / time_stamped('patient_counts.pickle')
    path_class_count = save_basedir / time_stamped('class_counts.pickle')
    with open(path_patient_count, 'wb') as f:
        pickle.dump(patient_counts, f)
    with open(path_class_count, 'wb') as f:
        pickle.dump(class_counts, f)


def count_patients(hdf5_info, patches_to_analyse):
    """
        count how many of the used patches were in each origin image used
        returns a dict with the image_ids of the origin images used as keys
        and number of patches from that image as value. Length is the num of img
    """
    # get infos of patch origin from pickle file and set the patch ids as index of the df
    hdf5_info = hdf5_info.set_index('patch id')

    counter = {}
    for patch_id in patches_to_analyse:
        patch_id = patch_id.decode() if isinstance(patch_id, bytes) else patch_id
        origin_image_id = hdf5_info.loc[patch_id, 'image number']
        if origin_image_id in counter:
            counter[origin_image_id] += 1
        else:
            counter[origin_image_id] = 1
    return counter


def count_classes(hdf5_file_path, patches_to_analyse):
    with h5py.File(hdf5_file_path, 'r') as hdf5_file:
        list_of_counts = []
        for patch_id in patches_to_analyse:
            label_data = hdf5_file['labels'][patch_id][()]
            num_pixels_per_class = np.add.reduce(label_data, axis=(0, 1, 2))
            list_of_counts.append(num_pixels_per_class)
    return list_of_counts


# Following function from
# https://stackoverflow.com/questions/16713643/how-to-add-the-date-to-the-file-name
def time_stamped(fname, fmt='%Y-%m-%d-%H-%M-{fname}'):
    # This creates a timestamped filename so we don't overwrite our good work
    return datetime.datetime.now().strftime(fmt).format(fname=fname)


# run run() if file executed as main
if __name__ == '__main__':
    run()
