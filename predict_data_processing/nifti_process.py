import nibabel as nib
import numpy as np
import scipy.io as sio
import yaml
import argparse

def nifti_reorientation(data_array, init_axcodes, final_axcodes):
    """
    Reorientation of nifti image
    :param data_array: type ndarray, image data
    :param init_axcodes: type tuple of 3 characters, original orientation. e.g. ('R','A', 'S')
    :param final_axcodes: type tuple of 3 characters, aimed orientation. e.g. ('P','L', 'S')
    :return: type ndarray: result image data
    """
    # orientation initial
    ornt_init = nib.orientations.axcodes2ornt(init_axcodes)
    # orientation final
    ornt_fin = nib.orientations.axcodes2ornt(final_axcodes)

    ornt_transf = nib.orientations.ornt_transform(ornt_init, ornt_fin)
    if np.array_equal(ornt_init, ornt_fin):
        return data_array
    return nib.orientations.apply_orientation(data_array, ornt_transf)


def read_nii_from_path(config, path=None):
    """
    Read nii file from the path
    :param config: type dict: config parameter
    :param path: path to nii.file
    :return: img_arr : type ndarray:  3D array from nii file
    :return: img.header: type nibabel.nifti1.Nifti1Header: header of nifti image
    :return: img.affine: type ndarray, affine info of nifti image
    """
    img = nib.load(path)
    axcodes = tuple(nib.aff2axcodes(img.affine))
    img_arr = img.get_fdata()
    if config['new_orientation']:
        img_arr = nifti_reorientation(np.array(img_arr), axcodes, tuple(config['new_orientation'])) #('P','L','S')
    if config['rescale']:
        slope = config['rescale'] / (np.max(img_arr) + 1e-16)
        img_arr = img_arr / (np.max(img_arr) + 1e-16) * config['rescale']
        img.header.set_slope_inter(slope, inter=0)
    elif config['scale']:
        img_arr = img_arr*config['scale']
    else:
        pass
    return img_arr, img.header, img.affine


def read_nifti(config_file, name_ID=None):
    """
    Read and process nifti image by config file and name_ID.

     :param config_file: type dict: main config parameter
     :param name_ID: type str: patient name ID
    :return: imgs_data: type ndarray, loaded nifti data
    """
    def _read(config, name, data):
        _, header, affine = read_nii_from_path(config, config['dict_name_path'][name][0])
        if config['img_path_per_channel']:
            # list_img_data, header, affine = [read_nii_path(config, path) for path in config['dict_name_path'][name]]
            list_img_data = [read_nii_from_path(config, path)[0] for path in config['dict_name_path'][name]]
            data = np.stack(list_img_data, axis=-1)
            img_data.append(data)
        else:
            data = read_nii_from_path(config, config['dict_name_path'][name])
            img_data.append(data)
        if config['save_mat_dir']:
            sio.savemat(config['save_mat_dir'] + '/' + name + '.mat',
                        {'img': data, 'header': header, 'affine': affine})
        elif config['save_nii_dir']:
            pass
        elif config['save_numpy_dir']:
            np.save({'img': data, 'header': header, 'affine': affine})
        return img_data

    config = yaml.safe_load(open(config_file['path_yaml_file']['predict_nifti'], "r").read())
    img_data = []
    if not name_ID:
        for name in config['dict_name_path']:
            img_data = _read(config, name, img_data)
    else:
        img_data = _read(config, name_ID, img_data)[0]
    return np.array(img_data)
