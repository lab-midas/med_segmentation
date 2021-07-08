import tensorflow as tf
import scipy.io as sio
import os
import numpy as np

def read_mat_file(path=None, channels_img=None,labels_name=None,dim=3,read_label=True, read_img=True,read_info=True,regularize_img=True):
    """
    Read mat file which contains image data and label data.
    Return images data and labels data with shape(x,y.z, channels)

    :param path: type str, mat file path
    :param imgs_name: type list str, names of channels which are saved in the mat file
    :param labels_name:  type str, img name which are saved in the mat file
    :param
    :return:imgs_data type ndarray, labels_data type ndarray, info: type dict

    """
    info=None

    return_list=[None,None,None]
    if channels_img is None:channels_img = ['img']
    if not isinstance(channels_img,list): channels_img =[channels_img]
    if labels_name is None: labels_name = ['P_BG', 'P_LT', 'P_VAT', 'P_AT']
    if not isinstance(labels_name, list): labels_name = [labels_name]

    patient_dict = sio.loadmat(path)
    # item : info or auxsave must be contained in the mat file.
    if read_info:
        info = patient_dict['info'] if 'info' in patient_dict.keys() else patient_dict['auxsave']
    if read_img:
        # imgs_data image as shape (x, y, z,channels)
        imgs_data = np.array([patient_dict[channel_img] for channel_img in channels_img])

        if regularize_img:
            imgs_data=imgs_data/np.max(imgs_data)*1 # 1 for scale



        # channel at last dimension
        if len(imgs_data.shape)==dim+1:
            imgs_data=np.rollaxis(imgs_data, 0, dim+1)
        #  If img_data is 5D, the first dimension is redundant, remove it.
        if len(imgs_data.shape)==dim+2:imgs_data=imgs_data[0]
        return_list[0]=np.float32(imgs_data)
    if read_label:
        # Read label data and set the last dimension to channels
        labels_data= np.array([patient_dict[label_name] for label_name in labels_name ])
        labels_data=np.rollaxis(labels_data, 0, dim+1)
        return_list[1]=np.float32(labels_data)

    if read_info:
        info_dict=dict()
        for info_key in info.dtype.names:
            info_dict[info_key] =info[0][0][info_key][0]
        return_list.append(info)

    return return_list[0],return_list[1],return_list[2]


def read_mat_file_body_identification(path=None, channels_img=None,labels_name=None,dim=3,read_img=True,read_label=True,read_info=True):
    """Especially for network body identification
        Read mat file which contains image data and label data.
    Return images data and labels data with shape(x,y.z, channels)

    :param path: type str, mat file path
    :param imgs_name: type list str, names of channels which are saved in the mat file
    :param labels_name:  type str, img name which are saved in the mat file
    :return:imgs_data type ndarray, labels_data type ndarray, info: type dict



    """
    info = None
    return_list = [None, None, None]
    if channels_img is None: channels_img = ['img']
    if labels_name is None: labels_name = ['hip', 'shoulder', 'heartEnd', 'heel','wrist']

    patient_dict = sio.loadmat(path)
    # item : info or auxsave must be contained in the mat file.
    if read_info:
        info = patient_dict['info'] if 'info' in patient_dict.keys() else patient_dict['auxsave']

    if read_img:
        # imgs_data image as shape (x, y, z,channels)
        imgs_data = np.array([patient_dict[channel_img] for channel_img in channels_img])

        # channel at last dimension
        if len(imgs_data.shape) == dim + 1:
            imgs_data = np.rollaxis(imgs_data, 0, dim + 1)
        #  If img_data is 5D, the first dimension is redundant, remove it.
        if len(imgs_data.shape) == dim + 2: imgs_data = imgs_data[0]
        return_list[0] = np.float32(imgs_data)
    if read_label:
        # Read label data and set the last dimension to channels
        # Get label position from item 'auxsave'
        labels_data = np.array([patient_dict['auxsave'][label_name][0][0][0][0] for label_name in labels_name])
        # Add label '0' at last for pipeline training.
        labels_data=np.array(np.concatenate((labels_data,[0]),axis=-1))
        return_list[1] = labels_data.astype(np.float32)

    if read_info:
        info_dict = dict()
        for info_key in info.dtype.names:
            info_dict[info_key] = info[0][0][info_key][0]
        return_list[2]=info

    return return_list[0], return_list[1], return_list[2]









