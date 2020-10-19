import os

import numpy as np
import nibabel as nib
from scipy.io import loadmat

root_mat_files = '/mnt/data/rawdata/NAKO/AT_NAKO'
mat_file_name = 'rework.mat'
image_size = [320, 260, 316]
path_save_labels = '/mnt/share/rahauei1/AT_Thomas/labels'
path_save_images = '/mnt/share/rahauei1/AT_Thomas/images'
convert_images = True  # this should be done on a large scale
# Use consecutive integers for class labels, 0 is considered background
labels = {'P_BG': 0,
          'P_AT': 1,
          'P_LT': 2,
          'P_VAT': 3}
sequences = {0: 'fat',
             1: 'water',
             2: 'in',
             3: 'opp'}
labels_already_processed = ['_'.join(i.split('_')[1:3]) for i in os.listdir(path_save_labels)]
# Iterate over subjects
for subject in os.listdir(root_mat_files):
    if subject == '103828_30':
        continue
    elif [subject for i in labels_already_processed if subject == i] is None:
        continue
    else:
        mat_content = loadmat(os.path.join(root_mat_files, subject, mat_file_name))
        # Initialize empty image array
        img = np.zeros(image_size)
        for seg_class in labels:
            seg_map = mat_content[seg_class]
            # Swap axes from height x width x depth to width x height x depth
            seg_map = np.swapaxes(seg_map, 0, 1)
            # Add weighted segmentation mask to initialized image
            img += seg_map * labels[seg_class]
        # Create NIFTI image
        nifti_image = nib.Nifti1Image(np.ushort(img), affine=None)
        # Save NIFTI image as path_save_labels/AT_subject.nii.gz
        nib.save(nifti_image, os.path.join(path_save_labels, f'AT_{subject}.nii.gz'))
        if convert_images:
            img_data = mat_content['img']
            img_data = np.swapaxes(img_data, 0, 1)
            for i in sequences:
                sequence_data = img_data[..., i]
                sequence_nifti = nib.Nifti1Image(np.ushort(sequence_data), affine=None)
                nib.save(sequence_nifti, os.path.join(path_save_images, f'AT_{subject}_{i:04d}.nii.gz'))
